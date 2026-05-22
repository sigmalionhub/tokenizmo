use thiserror::Error;
use crate::encoder::LEADING_SPACE_FLAG;
use crate::vocab::VocabTrie;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("unknown token id {0}")]
    UnknownToken(u32),
    #[error("decoded bytes are not valid UTF-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

pub struct Decoder {
    /// token_bytes[id] = raw bytes for that token
    token_bytes: Vec<Vec<u8>>,
}

impl Decoder {
    pub fn from_trie(trie: &VocabTrie) -> Self {
        Self {
            token_bytes: trie.token_bytes.clone(),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, DecodeError> {
        let mut buf: Vec<u8> = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let has_space = (id & LEADING_SPACE_FLAG) != 0;
            let base_id = (id & !LEADING_SPACE_FLAG) as usize;
            if base_id >= self.token_bytes.len() {
                return Err(DecodeError::UnknownToken(id));
            }
            if has_space {
                buf.push(b' ');
            }
            buf.extend_from_slice(&self.token_bytes[base_id]);
        }
        String::from_utf8(buf).map_err(DecodeError::Utf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::VocabTrie;

    #[test]
    fn decode_basic() {
        let mut trie = VocabTrie::new();
        let id = trie.insert(b"hello", -1.0);
        let dec = Decoder::from_trie(&trie);
        assert_eq!(dec.decode(&[id]).unwrap(), "hello");
    }

    #[test]
    fn decode_leading_space_flag() {
        let mut trie = VocabTrie::new();
        let id = trie.insert(b"world", -1.0);
        let dec = Decoder::from_trie(&trie);
        let flagged = id | LEADING_SPACE_FLAG;
        assert_eq!(dec.decode(&[flagged]).unwrap(), " world");
    }

    #[test]
    fn unknown_token_error() {
        let trie = VocabTrie::new();
        let dec = Decoder::from_trie(&trie);
        assert!(dec.decode(&[9999]).is_err());
    }
}
