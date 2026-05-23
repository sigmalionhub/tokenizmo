use std::sync::Arc;
use crate::vocab::VocabTrie;

/// Max token byte length considered during encoding (must match training).
const MAX_TOKEN_LEN: usize = 32;

/// Bit-flag: set on token ID when the token is preceded by a space.
pub const LEADING_SPACE_FLAG: u32 = 1 << 22;

/// DP state for Viterbi encoding.
#[derive(Clone, Copy)]
pub struct DpEntry {
    pub count: u32,
    pub neg_log_prob: f32,
    pub prev_pos: u32,
    pub token_id: u32,
}

impl DpEntry {
    pub const UNREACHABLE: Self = Self {
        count: u32::MAX,
        neg_log_prob: f32::INFINITY,
        prev_pos: 0,
        token_id: u32::MAX,
    };

    #[inline(always)]
    fn is_better_than(&self, other: &Self) -> bool {
        self.count < other.count
            || (self.count == other.count && self.neg_log_prob < other.neg_log_prob)
    }
}

pub struct Encoder {
    pub trie: Arc<VocabTrie>,
}

impl Encoder {
    pub fn new(mut trie: Arc<VocabTrie>) -> Self {
        if let Some(t) = Arc::get_mut(&mut trie) {
            t.finalize();
        }
        assert!(
            !trie.darts.is_empty(),
            "Encoder: call trie.finalize() before wrapping in Arc when sharing with other owners"
        );
        Self { trie }
    }

    /// Viterbi DP hot path with DARTS AoS lookup and software prefetch.
    ///
    /// # Cache behaviour
    ///
    /// DARTS AoS layout: `darts[node]` is 16 bytes = `{base, check, token, lp}`.
    /// Loading it once gives the base for child dispatch AND terminal info.
    ///
    /// In the inner loop, iteration j loads `darts[node]` and `darts[t]`.
    /// In iteration j+1, `node = t`, so `darts[node]` is **already in L1 cache**.
    /// Only one new cache miss per iteration (the child load `darts[t]`).
    ///
    /// The `_mm_prefetch` hint is issued for `darts[t]` before we need it,
    /// hiding the remaining LLC→L1 latency behind other in-flight work.
    pub fn encode_into(&self, text: &str, dp_buf: &mut Vec<DpEntry>, out: &mut Vec<u32>) {
        let trie = &*self.trie;
        let bytes = text.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return;
        }

        dp_buf.clear();
        dp_buf.resize(n + 1, DpEntry::UNREACHABLE);
        dp_buf[0] = DpEntry { count: 0, neg_log_prob: 0.0, prev_pos: 0, token_id: 0 };

        // Precompute root → ' ' transition once (reused in every leading-space check).
        let space_child = trie.darts_next(0, b' ');
        // Cache the space child's terminal info to avoid re-loading inside the loop.
        let space_tid_lp = if space_child != u32::MAX {
            trie.darts_token_lp(space_child)
        } else {
            None
        };

        let darts_ptr = trie.darts.as_ptr();
        let darts_len = trie.darts.len() as u32;
        let long_map = &trie.long_token_map;

        for i in 0..n {
            // SAFETY: i < n, dp_buf.len() == n+1
            let dp_i = unsafe { *dp_buf.get_unchecked(i) };
            if dp_i.count == u32::MAX {
                continue;
            }

            let byte_i = unsafe { *bytes.get_unchecked(i) };
            let (scan_start, leading_space) = if byte_i == b' ' && i + 1 < n {
                (i + 1, true)
            } else {
                (i, false)
            };

            let base_count = dp_i.count.wrapping_add(1);
            let base_neg_lp = dp_i.neg_log_prob;
            let base_prev = i as u32;
            let space_flag: u32 = if leading_space { LEADING_SPACE_FLAG } else { 0 };

            // ── DARTS inner walk (depth 0..darts_depth_limit) ────────────────
            //
            // SAFETY invariants maintained throughout:
            //   • `node` starts at 0 (valid) and only advances via validated DARTS transitions.
            //   • `t` is bounds-checked before any dereference.
            //   • `j` ∈ scan_start..n ⊆ 0..bytes.len().
            //   • `end = j+1 ≤ n`, dp_buf.len() == n+1.
            let depth_limit = trie.darts_depth_limit;
            let mut node = 0u32;
            let mut darts_depth = 0usize;
            for j in scan_start..n {
                // Hard limit: DARTS only covers the first `depth_limit` bytes.
                if darts_depth >= depth_limit {
                    break;
                }

                let byte = unsafe { *bytes.get_unchecked(j) };

                // Load current-node entry (L1 cache hit from previous iter, except first).
                let entry = unsafe { *darts_ptr.add(node as usize) };
                let t = entry.base.wrapping_add(byte as u32);

                if t >= darts_len {
                    break;
                }

                // Issue prefetch for the child entry before doing other work.
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                    _mm_prefetch(darts_ptr.add(t as usize) as *const i8, _MM_HINT_T0);
                }

                // Load child and validate parent check.
                let child = unsafe { *darts_ptr.add(t as usize) };
                if child.check != node {
                    break;
                }
                node = t;
                darts_depth += 1;

                // Terminal: update DP if this candidate is better.
                if child.token != u32::MAX {
                    let candidate = DpEntry {
                        count: base_count,
                        neg_log_prob: base_neg_lp - child.lp,
                        prev_pos: base_prev,
                        token_id: child.token | space_flag,
                    };
                    let slot = unsafe { dp_buf.get_unchecked_mut(j + 1) };
                    if candidate.is_better_than(slot) {
                        *slot = candidate;
                    }
                }
            }

            // ── HashMap fallback for long tokens ──────────────────────────────
            //
            // Only triggered when DARTS reached full depth_limit, which guarantees
            // the first depth_limit bytes form a valid trie prefix — a necessary
            // condition for any long token to match here.
            if darts_depth >= depth_limit && !long_map.is_empty() {
                let max_end = (scan_start + MAX_TOKEN_LEN).min(n);
                for end in (scan_start + depth_limit + 1)..=max_end {
                    // SAFETY: scan_start and end are within 0..n (bounds checked above).
                    let tok = unsafe { bytes.get_unchecked(scan_start..end) };
                    if let Some(&(tid, lp)) = long_map.get(tok) {
                        let candidate = DpEntry {
                            count: base_count,
                            neg_log_prob: base_neg_lp - lp,
                            prev_pos: base_prev,
                            token_id: tid | space_flag,
                        };
                        let slot = unsafe { dp_buf.get_unchecked_mut(end) };
                        if candidate.is_better_than(slot) {
                            *slot = candidate;
                        }
                    }
                }
            }

            // Try space as standalone token (leading_space case).
            if leading_space {
                if let Some((tid, lp)) = space_tid_lp {
                    let candidate = DpEntry {
                        count: base_count,
                        neg_log_prob: base_neg_lp - lp,
                        prev_pos: base_prev,
                        token_id: tid,
                    };
                    let slot = unsafe { dp_buf.get_unchecked_mut(i + 1) };
                    if candidate.is_better_than(slot) {
                        *slot = candidate;
                    }
                }
            }
        }

        // ── Backtrack ────────────────────────────────────────────────────────
        if dp_buf[n].count == u32::MAX {
            // Every byte must be reachable via add_guaranteed_tokens; if we get
            // here the vocabulary is missing coverage (binary input or broken
            // vocab).  Panic loudly rather than silently emitting wrong IDs.
            panic!(
                "encode: text not fully reachable — \
                 vocabulary missing coverage for input of {} bytes",
                n
            );
        }

        let token_count = dp_buf[n].count as usize;
        let start_idx = out.len();
        out.resize(start_idx + token_count, 0);

        let mut pos = n;
        let mut write_idx = start_idx + token_count;
        while pos > 0 {
            let entry = unsafe { *dp_buf.get_unchecked(pos) };
            write_idx -= 1;
            unsafe { *out.get_unchecked_mut(write_idx) = entry.token_id; }
            pos = entry.prev_pos as usize;
        }
    }

    fn encode_chunk(&self, text: &str) -> Vec<u32> {
        use std::cell::RefCell;
        thread_local! {
            static DP_BUF: RefCell<Vec<DpEntry>> = RefCell::new(Vec::new());
        }
        DP_BUF.with(|cell| {
            let mut dp_buf = cell.borrow_mut();
            let mut out = Vec::new();
            self.encode_into(text, &mut *dp_buf, &mut out);
            out
        })
    }

    /// Encode `text` into token IDs.
    ///
    /// Inputs shorter than `PARALLEL_THRESHOLD` bytes are encoded on the
    /// calling thread.  Longer inputs are split into word-level chunks and
    /// processed in parallel via Rayon.
    ///
    /// The split keeps spaces at the **start** of the following chunk so that
    /// `encode_into` applies `LEADING_SPACE_FLAG` correctly — producing the
    /// same token IDs as encoding the whole string in one pass.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Below threshold: single-thread is faster (no Rayon spawn overhead).
        const PARALLEL_THRESHOLD: usize = 1024;
        // Accumulate words into a chunk until it reaches this size, then split.
        const MIN_CHUNK_BYTES: usize = 256;

        let bytes = text.as_bytes();
        if bytes.len() < PARALLEL_THRESHOLD {
            return self.encode_chunk(text);
        }

        use rayon::prelude::*;
        split_words(bytes, MIN_CHUNK_BYTES)
            .into_par_iter()
            .flat_map_iter(|chunk| {
                // SAFETY: split_words only splits at ASCII 0x20 boundaries,
                // which are always valid UTF-8 code-unit positions.
                let s = unsafe { std::str::from_utf8_unchecked(chunk) };
                self.encode_chunk(s)
            })
            .collect()
    }
}

/// Split `bytes` into chunks at word boundaries for parallel encoding.
///
/// Spaces are grouped with the *following* word so `encode_into` sees the
/// leading space and sets `LEADING_SPACE_FLAG` correctly — producing
/// identical token IDs to encoding the whole string in one pass.
///
/// Words accumulate into a chunk until it reaches `min_bytes`; the split
/// then falls on the next word boundary so no token ever spans a chunk edge.
fn split_words(bytes: &[u8], min_bytes: usize) -> Vec<&[u8]> {
    let n = bytes.len();
    let mut chunks = Vec::new();
    let mut chunk_start = 0;
    let mut i = 0;

    while i < n {
        // Consume optional leading spaces then a non-space word.
        while i < n && bytes[i] == b' ' { i += 1; }
        while i < n && bytes[i] != b' ' { i += 1; }
        // `i` is now at a space or end-of-string — a clean word boundary.
        // Emit a chunk once it has accumulated enough bytes, but only when
        // there is still more text (so the last chunk is never emitted here).
        if i - chunk_start >= min_bytes && i < n {
            chunks.push(&bytes[chunk_start..i]);
            chunk_start = i;
        }
    }

    if chunk_start < n {
        chunks.push(&bytes[chunk_start..n]);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabTrie;

    fn make_encoder(tokens: &[(&[u8], f32)]) -> Encoder {
        let mut trie = VocabTrie::new();
        for (tok, lp) in tokens {
            trie.insert(tok, *lp);
        }
        Encoder::new(Arc::new(trie))
    }

    /// Build encoder with full DARTS (no HashMap) for use as reference in tests.
    fn make_encoder_full_darts(tokens: &[(&[u8], f32)]) -> Encoder {
        let mut trie = VocabTrie::new();
        for (tok, lp) in tokens {
            trie.insert(tok, *lp);
        }
        trie.finalize_with_depth(usize::MAX);
        Encoder { trie: Arc::new(trie) }
    }

    /// Build encoder with hybrid DARTS+HashMap using given depth limit.
    fn make_encoder_hybrid(tokens: &[(&[u8], f32)], depth: usize) -> Encoder {
        let mut trie = VocabTrie::new();
        for (tok, lp) in tokens {
            trie.insert(tok, *lp);
        }
        trie.finalize_with_depth(depth);
        Encoder { trie: Arc::new(trie) }
    }

    #[test]
    fn simple_encode() {
        let enc = make_encoder(&[
            (b"hello", -1.0),
            (b" ", -0.5),
            (b"world", -1.0),
        ]);
        let ids = enc.encode("hello world");
        assert!(!ids.is_empty());
    }

    #[test]
    fn prefers_fewer_tokens() {
        let enc = make_encoder(&[
            (b"a", -1.0),
            (b"b", -1.0),
            (b"ab", -1.5),
        ]);
        let ids = enc.encode("ab");
        assert_eq!(ids.len(), 1);
        let ab_id = enc.trie.get(b"ab").unwrap();
        assert_eq!(ids[0], ab_id);
    }

    #[test]
    fn encode_empty() {
        let enc = make_encoder(&[(b"a", -1.0)]);
        assert_eq!(enc.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn round_trip_leading_space() {
        let enc = make_encoder(&[
            (b"hello", -1.0),
            (b"world", -1.0),
            (b" ", -0.5),
            (b" world", -0.8),
        ]);
        let ids = enc.encode("hello world");
        assert!(!ids.is_empty());
        for &id in &ids {
            let base_id = id & !(LEADING_SPACE_FLAG);
            assert!((base_id as usize) < enc.trie.vocab_size);
        }
    }

    /// Validation: hybrid DARTS+HashMap must produce identical token sequences
    /// to the full DARTS encoder.  Tests short tokens, long tokens (> threshold),
    /// tokens that share prefixes, and leading-space tokens.
    #[test]
    fn hybrid_matches_full_darts() {
        // Build a vocab mixing short (≤ 8 bytes) and long (> 8 bytes) tokens.
        // Short: "a", " ", "the", "hello", "world", "12345678" (8 bytes)
        // Long:  "123456789" (9B), "tokenized" (9B), "representation" (14B)
        // Single-byte fallbacks cover all ASCII so the Viterbi never hits the
        // unreachable-text panic (required since the vocab has no per-byte coverage).
        let mut tokens_vec: Vec<(&[u8], f32)> = vec![
            (b"a",               -2.0),
            (b" ",               -0.5),
            (b"the",             -1.0),
            (b"hello",           -1.2),
            (b"world",           -1.2),
            (b"12345678",        -1.5),   // exactly 8 bytes → DARTS
            (b"123456789",       -1.4),   // 9 bytes → HashMap
            (b"tokenized",       -1.3),   // 9 bytes → HashMap
            (b"representation",  -1.6),   // 14 bytes → HashMap
            (b"token",           -1.1),   // 5 bytes → DARTS (prefix of "tokenized")
        ];
        // Low-probability single-byte fallbacks for full ASCII coverage.
        let fallback_bytes: Vec<Vec<u8>> = (0u8..=127)
            .filter(|b| !matches!(b, b'a' | b' '))
            .map(|b| vec![b])
            .collect();
        for fb in &fallback_bytes {
            tokens_vec.push((fb.as_slice(), -10.0));
        }
        let tokens = tokens_vec.as_slice();

        let texts = [
            "hello world",
            "the representation",
            "123456789",
            "tokenized",
            "a the a",
            "12345678 tokenized representation hello",
            // Longer text to stress Viterbi DP with mixed short+long tokens
            "hello the tokenized representation of 12345678 and 123456789",
        ];

        let full = make_encoder_full_darts(tokens);
        let hybrid = make_encoder_hybrid(tokens, 8);

        // Sanity: hybrid has long tokens in HashMap, not in DARTS
        assert_eq!(hybrid.trie.long_token_count(), 3,
            "expected 3 long tokens (123456789, tokenized, representation)");

        for text in &texts {
            let ids_full   = full.encode(text);
            let ids_hybrid = hybrid.encode(text);
            assert_eq!(
                ids_full, ids_hybrid,
                "mismatch for {:?}\n  full:   {:?}\n  hybrid: {:?}",
                text, ids_full, ids_hybrid
            );
        }
    }

    /// Edge case: text that uses ONLY long tokens (> threshold).
    #[test]
    fn hybrid_long_token_only_text() {
        let tokens: &[(&[u8], f32)] = &[
            (b"a",              -3.0),  // fallback char
            (b" ",              -0.5),
            (b"abcdefghi",      -1.0),  // 9 bytes → HashMap
            (b"abcdefghij",     -0.9),  // 10 bytes → HashMap (better than 9+a)
        ];
        let full   = make_encoder_full_darts(tokens);
        let hybrid = make_encoder_hybrid(tokens, 8);

        let text = "abcdefghij";
        let ids_full   = full.encode(text);
        let ids_hybrid = hybrid.encode(text);
        assert_eq!(ids_full, ids_hybrid,
            "long-only mismatch: full={:?} hybrid={:?}", ids_full, ids_hybrid);
        // Should use the single 10-byte token, not the 9-byte one + "a"
        assert_eq!(ids_full.len(), 1, "expected 1 token for 'abcdefghij'");
    }

    /// Edge case: long token preceded by space (LEADING_SPACE_FLAG must be set).
    #[test]
    fn hybrid_long_token_with_leading_space() {
        let tokens: &[(&[u8], f32)] = &[
            (b"a",             -3.0),
            (b" ",             -0.5),
            (b"hello",         -1.0),
            (b"tokenized",     -1.0),  // 9 bytes → HashMap
        ];
        let full   = make_encoder_full_darts(tokens);
        let hybrid = make_encoder_hybrid(tokens, 8);

        let text = "hello tokenized";
        let full_ids   = full.encode(text);
        let hybrid_ids = hybrid.encode(text);
        assert_eq!(full_ids, hybrid_ids,
            "leading-space mismatch: full={:?} hybrid={:?}", full_ids, hybrid_ids);

        // The second token should have LEADING_SPACE_FLAG set
        assert!(hybrid_ids.len() >= 2);
        let second = hybrid_ids[1];
        assert!(second & LEADING_SPACE_FLAG != 0,
            "expected LEADING_SPACE_FLAG on 'tokenized' token, got id={second}");
    }
}
