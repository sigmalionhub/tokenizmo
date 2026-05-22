use thiserror::Error;

#[derive(Debug, Error)]
pub enum VocabError {
    #[error("invalid vocab file: {0}")]
    InvalidFormat(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// One slot in the Double-Array trie.  16 bytes = one 16-byte load.
///
/// Stored as **array-of-structs** so that a single load of `darts[node]`
/// delivers `base` (for child dispatch), `check` (for parent validation in the
/// *next* iteration's load), `token` and `lp` (terminal info) all in one shot.
///
/// Access pattern in the hot inner loop:
/// ```text
/// iter j  : entry = darts[node]        ← 1 cache miss
///           t     = entry.base + byte
///           child = darts[t]           ← 1 cache miss
///           valid  = child.check == node
///           node  = t  (if valid)
/// iter j+1: entry = darts[node = t]   ← L1 HIT  (already loaded above)
///           t2    = entry.base + byte2
///           child2 = darts[t2]         ← 1 cache miss
/// …
/// ```
///
/// With SoA, `base[node]` and `check[t]` live in different arrays, causing 2
/// independent cache misses per iteration.  AoS reduces this to 1 miss per
/// iteration (the child load `darts[t]`) because the current node's data is
/// already in cache from the previous iteration.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct DartsEntry {
    /// Base offset: child of this node via byte c is at index `base + c`.
    pub base: u32,
    /// Parent check: `darts[child].check` must equal the parent's DARTS index.
    pub check: u32,
    /// Token ID if this is a terminal node; `u32::MAX` otherwise.
    pub token: u32,
    /// Log-probability (valid only when `token != u32::MAX`).
    pub lp: f32,
}

/// Compact trie with Compressed Sparse Row (CSR) for correctness + Double-Array
/// Trie (DARTS) AoS layout for hot-path O(1) child lookup.
///
/// See [`DartsEntry`] for the cache-locality rationale.
pub struct VocabTrie {
    // ── CSR representation (kept for correctness / fallback) ────────────────
    child_start: Vec<u32>,
    children_bytes: Vec<u8>,
    children_ids: Vec<u32>,
    token_ids: Vec<u32>,

    // ── DARTS AoS representation (hot path) ─────────────────────────────────
    /// Flat array of DARTS slots.  Root is at index 0.  Slot 0's `check` is
    /// `u32::MAX` (sentinel: root has no parent).
    pub darts: Vec<DartsEntry>,

    // ── Build-time representation (cleared by finalize) ──────────────────────
    build_children: Vec<Vec<(u8, u32)>>,
    build_token_ids: Vec<u32>,
    num_nodes: usize,

    // ── Per-token data (used by vocab_io and the trainer) ───────────────────
    pub log_probs: Vec<f32>,
    pub token_bytes: Vec<Vec<u8>>,
    pub vocab_size: usize,

    finalized: bool,
}

impl VocabTrie {
    pub fn new() -> Self {
        Self {
            child_start: Vec::new(),
            children_bytes: Vec::new(),
            children_ids: Vec::new(),
            token_ids: Vec::new(),
            darts: Vec::new(),
            build_children: vec![Vec::new()],
            build_token_ids: vec![u32::MAX],
            num_nodes: 1,
            log_probs: Vec::new(),
            token_bytes: Vec::new(),
            vocab_size: 0,
            finalized: false,
        }
    }

    pub fn insert(&mut self, token: &[u8], log_prob: f32) -> u32 {
        debug_assert!(!self.finalized, "insert after finalize");
        let token_id = self.vocab_size as u32;
        self.log_probs.push(log_prob);
        self.token_bytes.push(token.to_vec());
        self.vocab_size += 1;

        let mut node = 0usize;
        for &byte in token {
            let pos = self.build_children[node]
                .binary_search_by_key(&byte, |&(b, _)| b);
            node = match pos {
                Ok(idx) => self.build_children[node][idx].1 as usize,
                Err(ins) => {
                    let new_id = self.num_nodes as u32;
                    self.num_nodes += 1;
                    self.build_children.push(Vec::new());
                    self.build_token_ids.push(u32::MAX);
                    self.build_children[node].insert(ins, (byte, new_id));
                    new_id as usize
                }
            };
        }
        self.build_token_ids[node] = token_id;
        token_id
    }

    /// Compact build-time representation into CSR, then build DARTS.
    pub fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        let n = self.num_nodes;
        let total: usize = self.build_children.iter().map(|c| c.len()).sum();

        self.child_start = Vec::with_capacity(n + 1);
        self.children_bytes = Vec::with_capacity(total);
        self.children_ids = Vec::with_capacity(total);
        self.token_ids = Vec::with_capacity(n);

        let mut offset = 0u32;
        for i in 0..n {
            self.child_start.push(offset);
            self.token_ids.push(self.build_token_ids[i]);
            for &(byte, child) in &self.build_children[i] {
                self.children_bytes.push(byte);
                self.children_ids.push(child);
                offset += 1;
            }
        }
        self.child_start.push(offset);

        self.build_children = Vec::new();
        self.build_token_ids = Vec::new();
        self.finalized = true;

        self.build_darts();
    }

    /// Build the DARTS AoS array from the finalized CSR.
    ///
    /// BFS assigns each CSR node a unique DARTS position.  Root → position 0.
    /// For each node at position p with children {(c, t), …}:
    ///   • Find smallest base b ≥ 1 where positions b+c are all free.
    ///   • Set `darts[p].base = b`.
    ///   • For each (c, t): mark child position b+c with `check = p`, copy
    ///     terminal info, enqueue (t, b+c).
    fn build_darts(&mut self) {
        let n = self.num_nodes;
        let mut cap = n.saturating_mul(4).max(512) + 512;

        let mut entries: Vec<DartsEntry> = vec![DartsEntry::default(); cap];
        let mut used: Vec<bool> = vec![false; cap];

        // Position 0 = root.  check[0] = u32::MAX (no parent, sentinel).
        entries[0].check = u32::MAX;
        used[0] = true;

        // Copy root's terminal info.
        let root_tid = self.token_ids[0];
        if root_tid != u32::MAX {
            entries[0].token = root_tid;
            entries[0].lp = self.log_probs[root_tid as usize];
        } else {
            entries[0].token = u32::MAX;
        }

        // BFS: (csr_node, darts_position)
        let mut queue: std::collections::VecDeque<(usize, usize)> =
            std::collections::VecDeque::with_capacity(n);
        queue.push_back((0, 0));

        while let Some((csr, pos)) = queue.pop_front() {
            let cs = self.child_start[csr] as usize;
            let ce = self.child_start[csr + 1] as usize;
            if cs == ce {
                continue; // leaf
            }

            let node_bytes = &self.children_bytes[cs..ce];
            let max_byte = node_bytes.last().copied().unwrap_or(0) as usize;

            let b = find_darts_base(&used, node_bytes);

            let needed = b + max_byte + 1;
            if needed > cap {
                cap = (needed + 512).max(cap * 2);
                entries.resize(cap, DartsEntry::default());
                used.resize(cap, false);
                // Re-init token field of new slots to u32::MAX.
                for e in &mut entries[needed - 1..cap] {
                    e.token = u32::MAX;
                }
            }

            entries[pos].base = b as u32;

            for (k, &c) in node_bytes.iter().enumerate() {
                let child_pos = b + c as usize;
                used[child_pos] = true;
                entries[child_pos].check = pos as u32;

                let csr_child = self.children_ids[cs + k] as usize;
                let tid = self.token_ids[csr_child];
                if tid != u32::MAX {
                    entries[child_pos].token = tid;
                    entries[child_pos].lp = self.log_probs[tid as usize];
                } else {
                    entries[child_pos].token = u32::MAX;
                }

                queue.push_back((csr_child, child_pos));
            }
        }

        // Trim trailing unused slots.
        let used_len = used.iter().rposition(|&u| u).map_or(1, |p| p + 1);
        entries.truncate(used_len);
        self.darts = entries;
    }

    /// O(1) child lookup via DARTS.  Returns child DARTS index or `u32::MAX`.
    ///
    /// # Safety
    /// `node` must be a valid DARTS index (< darts.len()).
    #[inline(always)]
    pub fn darts_next(&self, node: u32, byte: u8) -> u32 {
        // Load current node entry — covers base, check, token, lp in 16 bytes.
        let entry = unsafe { *self.darts.get_unchecked(node as usize) };
        let t = entry.base.wrapping_add(byte as u32);
        let len = self.darts.len() as u32;
        if t < len {
            // Load child entry — its check must equal our index.
            let child = unsafe { *self.darts.get_unchecked(t as usize) };
            if child.check == node { t } else { u32::MAX }
        } else {
            u32::MAX
        }
    }

    /// Returns `(token_id, log_prob)` if DARTS slot `t` is a terminal.
    ///
    /// # Safety
    /// `t` must be a valid DARTS index (< darts.len()).
    #[inline(always)]
    pub fn darts_token_lp(&self, t: u32) -> Option<(u32, f32)> {
        // SAFETY: caller guarantees t is a valid DARTS index.
        let e = unsafe { *self.darts.get_unchecked(t as usize) };
        if e.token != u32::MAX { Some((e.token, e.lp)) } else { None }
    }

    // ── CSR helpers (used by get(), vocab_io, and older tests) ──────────────

    #[inline(always)]
    pub fn child(&self, node: u32, byte: u8) -> u32 {
        if self.finalized {
            let start = self.child_start[node as usize] as usize;
            let end = self.child_start[node as usize + 1] as usize;
            let bytes = &self.children_bytes[start..end];
            match bytes.binary_search(&byte) {
                Ok(idx) => self.children_ids[start + idx],
                Err(_) => 0,
            }
        } else {
            let children = &self.build_children[node as usize];
            match children.binary_search_by_key(&byte, |&(b, _)| b) {
                Ok(idx) => children[idx].1,
                Err(_) => 0,
            }
        }
    }

    #[inline(always)]
    pub fn token_id_at(&self, node: u32) -> Option<u32> {
        let id = if self.finalized {
            self.token_ids[node as usize]
        } else {
            self.build_token_ids[node as usize]
        };
        if id == u32::MAX { None } else { Some(id) }
    }

    pub fn get(&self, token: &[u8]) -> Option<u32> {
        let mut node = 0u32;
        for &byte in token {
            node = self.child(node, byte);
            if node == 0 {
                return None;
            }
        }
        self.token_id_at(node)
    }
}

/// Find the smallest base offset b ≥ 1 where every position b + c is free.
fn find_darts_base(used: &[bool], bytes: &[u8]) -> usize {
    'outer: for b in 1usize.. {
        for &c in bytes {
            let pos = b + c as usize;
            if pos < used.len() && used[pos] {
                continue 'outer;
            }
        }
        return b;
    }
    unreachable!()
}

impl Default for VocabTrie {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_lookup() {
        let mut trie = VocabTrie::new();
        let id = trie.insert(b"hello", -1.0);
        assert_eq!(trie.get(b"hello"), Some(id));
        assert_eq!(trie.get(b"hell"), None);
        assert_eq!(trie.get(b"hello world"), None);
    }

    #[test]
    fn overlapping_prefixes() {
        let mut trie = VocabTrie::new();
        let id_h = trie.insert(b"he", -1.0);
        let id_hel = trie.insert(b"hello", -2.0);
        assert_eq!(trie.get(b"he"), Some(id_h));
        assert_eq!(trie.get(b"hello"), Some(id_hel));
    }

    #[test]
    fn insert_and_lookup_after_finalize() {
        let mut trie = VocabTrie::new();
        let id = trie.insert(b"hello", -1.0);
        trie.finalize();
        assert_eq!(trie.get(b"hello"), Some(id));
        assert_eq!(trie.get(b"hell"), None);
    }

    #[test]
    fn double_finalize_is_idempotent() {
        let mut trie = VocabTrie::new();
        trie.insert(b"x", -1.0);
        trie.finalize();
        trie.finalize();
        assert!(trie.get(b"x").is_some());
    }

    #[test]
    fn darts_matches_csr() {
        let mut trie = VocabTrie::new();
        trie.insert(b"hello", -1.0);
        trie.insert(b"hell", -0.5);
        trie.insert(b"he", -0.3);
        trie.insert(b"world", -1.2);
        trie.insert(b" ", -0.1);
        trie.finalize();

        let mut node = 0u32;
        for &b in b"hello" {
            let next = trie.darts_next(node, b);
            assert_ne!(next, u32::MAX, "darts_next failed at byte {b}");
            node = next;
        }
        let (tid, _) = trie.darts_token_lp(node).expect("hello should be terminal");
        assert_eq!(Some(tid), trie.get(b"hello"));

        assert_eq!(trie.darts_next(0, b'z'), u32::MAX);
    }

    #[test]
    fn darts_complete_vocab() {
        let tokens: &[(&[u8], f32)] = &[
            (b"the", -1.0),
            (b"th", -1.5),
            (b"t", -2.0),
            (b"he", -1.8),
            (b"h", -2.5),
            (b"e", -2.8),
            (b" the", -0.8),
            (b" ", -0.5),
        ];
        let mut trie = VocabTrie::new();
        for (tok, lp) in tokens {
            trie.insert(tok, *lp);
        }
        trie.finalize();

        for (tok, _) in tokens {
            let csr_id = trie.get(tok).expect("CSR must find token");

            let mut node = 0u32;
            let mut ok = true;
            for &b in *tok {
                let next = trie.darts_next(node, b);
                if next == u32::MAX { ok = false; break; }
                node = next;
            }
            assert!(ok, "DARTS could not traverse {:?}", tok);
            let (darts_id, darts_lp) = trie.darts_token_lp(node)
                .expect("DARTS must find terminal");
            assert_eq!(darts_id, csr_id);
            let expected_lp = trie.log_probs[csr_id as usize];
            assert!((darts_lp - expected_lp).abs() < 1e-6);
        }
    }
}
