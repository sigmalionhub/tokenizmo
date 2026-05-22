use rayon::prelude::*;
use std::cell::RefCell;
use crate::encoder::{Encoder, DpEntry};

thread_local! {
    static DP_BUF: RefCell<Vec<DpEntry>> = RefCell::new(Vec::new());
}

/// Encode a batch of texts in parallel using Rayon.
/// Each worker thread reuses its own DP scratch buffer (no allocations per call).
pub fn encode_batch(encoder: &Encoder, texts: &[&str]) -> Vec<Vec<u32>> {
    texts
        .par_iter()
        .map(|text| {
            DP_BUF.with(|cell| {
                let mut dp_buf = cell.borrow_mut();
                let mut out = Vec::new();
                encoder.encode_into(text, &mut *dp_buf, &mut out);
                out
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabTrie;
    use std::sync::Arc;

    #[test]
    fn batch_matches_sequential() {
        let mut trie = VocabTrie::new();
        trie.insert(b"hello", -1.0);
        trie.insert(b"world", -1.0);
        trie.insert(b" ", -0.5);
        let enc = Encoder::new(Arc::new(trie));

        let texts = vec!["hello", "world", "hello world"];
        let batch = encode_batch(&enc, &texts);
        for (text, ids) in texts.iter().zip(batch.iter()) {
            let seq = enc.encode(text);
            assert_eq!(ids, &seq, "mismatch for {:?}", text);
        }
    }
}
