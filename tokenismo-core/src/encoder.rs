use std::sync::Arc;
use crate::vocab::VocabTrie;

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

            // ── DARTS inner walk ───────────────────────────────────────────────
            //
            // Inlined to enable: (a) reading both fields from one cache line
            // without a function-call boundary, and (b) placing the prefetch
            // hint immediately after computing `t` for maximum lead time.
            //
            // SAFETY invariants maintained throughout:
            //   • `node` starts at 0 (valid) and only advances via validated DARTS transitions.
            //   • `t` is bounds-checked before any dereference.
            //   • `j` ∈ scan_start..n ⊆ 0..bytes.len().
            //   • `end = j+1 ≤ n`, dp_buf.len() == n+1.
            let mut node = 0u32;
            for j in scan_start..n {
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
            for &byte in bytes {
                out.push(byte as u32);
            }
            return;
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
    /// Inputs < `PARALLEL_THRESHOLD` bytes are encoded on the calling thread.
    /// Larger inputs are split at space boundaries into `CHUNK_SIZE`-byte
    /// chunks and processed in parallel via Rayon.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        const PARALLEL_THRESHOLD: usize = 4 * 1024; // 4 KB — lower than before
        const CHUNK_SIZE: usize = 4 * 1024;

        if text.len() < PARALLEL_THRESHOLD {
            return self.encode_chunk(text);
        }

        use rayon::prelude::*;
        split_at_spaces(text.as_bytes(), CHUNK_SIZE)
            .into_par_iter()
            .flat_map_iter(|chunk| self.encode_chunk(chunk))
            .collect()
    }
}

fn split_at_spaces(bytes: &[u8], chunk_size: usize) -> Vec<&str> {
    let n = bytes.len();
    let mut chunks = Vec::with_capacity(n / chunk_size + 1);
    let mut start = 0;

    while start < n {
        let mut end = (start + chunk_size).min(n);
        if end < n {
            if let Some(pos) = bytes[start + 1..end]
                .iter()
                .rposition(|&b| b == b' ')
            {
                end = start + 1 + pos + 1;
            } else {
                while end > start && (bytes[end] & 0xC0) == 0x80 {
                    end -= 1;
                }
            }
        }
        // SAFETY: start and end are at valid UTF-8 boundaries.
        chunks.push(unsafe { std::str::from_utf8_unchecked(&bytes[start..end]) });
        start = end;
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
        assert_eq!(enc.encode(""), vec![]);
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
}
