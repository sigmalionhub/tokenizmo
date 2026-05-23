use std::io::{Read, Write, BufWriter};
use std::path::Path;
use crate::vocab::{VocabTrie, VocabError};
use crate::trainer::TrainVocab;

const MAGIC: &[u8; 4] = b"NXT1";

/// Serialize vocab to binary format:
/// [magic:4][vocab_size:4u32][entries...]
/// Each entry: [token_len:1u8][token_bytes:N][log_prob:4f32]
pub fn save_vocab(trie: &VocabTrie, path: &Path) -> Result<(), VocabError> {
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(MAGIC)?;
    w.write_all(&(trie.vocab_size as u32).to_le_bytes())?;
    for id in 0..trie.vocab_size {
        let bytes = &trie.token_bytes[id];
        let log_prob = trie.log_probs[id];
        w.write_all(&[bytes.len() as u8])?;
        w.write_all(bytes)?;
        w.write_all(&log_prob.to_le_bytes())?;
    }
    Ok(())
}

/// Load vocab from binary format produced by `save_vocab`.
pub fn load_vocab(path: &Path) -> Result<VocabTrie, VocabError> {
    load_vocab_with_depth(path, usize::MAX)
}

/// Load vocab and finalize with an explicit DARTS depth limit.
///
/// Use `usize::MAX` for full DARTS (all tokens in trie, no HashMap fallback).
/// Use `SHORT_TOKEN_THRESHOLD` (the default) for the hybrid DARTS+HashMap layout.
pub fn load_vocab_with_depth(path: &Path, depth_limit: usize) -> Result<VocabTrie, VocabError> {
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(VocabError::InvalidFormat("bad magic bytes".into()));
    }
    let mut size_buf = [0u8; 4];
    file.read_exact(&mut size_buf)?;
    let vocab_size = u32::from_le_bytes(size_buf) as usize;

    let mut trie = VocabTrie::new();
    let mut lp_buf = [0u8; 4];
    for _ in 0..vocab_size {
        let mut len_buf = [0u8; 1];
        file.read_exact(&mut len_buf)?;
        let tok_len = len_buf[0] as usize;
        let mut tok = vec![0u8; tok_len];
        file.read_exact(&mut tok)?;
        file.read_exact(&mut lp_buf)?;
        let log_prob = f32::from_le_bytes(lp_buf);
        trie.insert(&tok, log_prob);
    }
    trie.finalize_with_depth(depth_limit);
    Ok(trie)
}

/// Save a training vocab (from the trainer) to the binary .vocab format.
/// The order of tokens in TrainVocab becomes the token ID assignment in the file.
pub fn save_vocab_from_train(vocab: &TrainVocab, path: &Path) -> Result<(), VocabError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(MAGIC)?;
    w.write_all(&(vocab.len() as u32).to_le_bytes())?;
    for entry in vocab.iter() {
        if entry.token.len() > 255 {
            return Err(VocabError::InvalidFormat(format!(
                "token too long ({} bytes): {:?}",
                entry.token.len(),
                entry.token
            )));
        }
        w.write_all(&[entry.token.len() as u8])?;
        w.write_all(&entry.token)?;
        w.write_all(&entry.log_prob.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_save_load() {
        let mut trie = VocabTrie::new();
        trie.insert(b"hello", -1.5);
        trie.insert(b"world", -2.0);
        trie.insert(b" ", -0.5);

        let dir = std::env::temp_dir();
        let path = dir.join("tokenismo_test_vocab.bin");
        save_vocab(&trie, &path).unwrap();
        let loaded = load_vocab(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.vocab_size, trie.vocab_size);
        assert_eq!(loaded.get(b"hello"), trie.get(b"hello"));
        assert_eq!(loaded.get(b"world"), trie.get(b"world"));
        assert_eq!(loaded.get(b" "), trie.get(b" "));
    }
}
