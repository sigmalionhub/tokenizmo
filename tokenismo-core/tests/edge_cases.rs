use std::sync::Arc;
use tokenismo_core::{Decoder, Encoder, VocabTrie};

/// Build an encoder + decoder covering all 256 byte values as single-byte tokens
/// plus a handful of multi-byte tokens for realistic coverage.
fn make_full_encoder() -> (Encoder, Decoder) {
    let mut trie = VocabTrie::new();
    for b in 0u8..=255 {
        trie.insert(&[b], -8.0);
    }
    trie.insert(b"hello", -1.0);
    trie.insert(b"world", -1.0);
    trie.insert(b" world", -1.5);
    trie.insert("Привет".as_bytes(), -1.0);
    trie.insert("мир".as_bytes(), -1.0);
    trie.insert("你好".as_bytes(), -1.0);
    trie.insert("😀".as_bytes(), -1.0);
    trie.finalize(); // must finalize before sharing via Arc
    let trie = Arc::new(trie);
    (Encoder::new(Arc::clone(&trie)), Decoder::from_trie(&trie))
}

fn roundtrip(text: &str) -> bool {
    let (enc, dec) = make_full_encoder();
    let ids = enc.encode(text);
    match dec.decode(&ids) {
        Ok(decoded) => decoded == text,
        Err(_) => false,
    }
}

#[test]
fn empty_string_returns_empty() {
    let (enc, dec) = make_full_encoder();
    let ids = enc.encode("");
    assert_eq!(ids, Vec::<u32>::new(), "empty string must produce no tokens");
    assert_eq!(dec.decode(&[]).unwrap(), "");
}

#[test]
fn single_space() {
    assert!(roundtrip(" "), "single space must round-trip");
    let (enc, _) = make_full_encoder();
    let ids = enc.encode(" ");
    assert_eq!(ids.len(), 1, "single space should be one token");
}

#[test]
fn nul_byte() {
    assert!(roundtrip("\0"), "NUL byte must round-trip");
}

#[test]
fn four_byte_emoji() {
    assert!(roundtrip("😀"), "4-byte emoji must round-trip");
    assert!(roundtrip("🎉🔥💡"), "emoji sequence must round-trip");
}

#[test]
fn max_length_token_boundary() {
    // 32 printable ASCII chars — exercise max_token_len boundary
    let s = "abcdefghijklmnopqrstuvwxyzabcdef";
    assert_eq!(s.len(), 32);
    assert!(roundtrip(s));
}

#[test]
fn whitespace_only() {
    assert!(roundtrip("   \t\n  "), "whitespace-only must not panic");
    assert!(roundtrip("\n\n\n"), "newlines only must not panic");
}

#[test]
fn one_mb_adversarial() {
    // 1 MB of repeated 'a' — exercises O(n) heap DP, must not stack overflow
    let big = "a".repeat(1_000_000);
    let (enc, dec) = make_full_encoder();
    let ids = enc.encode(&big);
    let decoded = dec.decode(&ids).unwrap();
    assert_eq!(decoded, big, "1 MB adversarial input must round-trip");
}

#[test]
fn mixed_scripts() {
    let mixed = "Hello мир 你好 مرحبا";
    assert!(roundtrip(mixed), "mixed script string must round-trip");
}

#[test]
fn crlf_line_endings() {
    assert!(roundtrip("line1\r\nline2\r\n"), "CRLF must round-trip");
}

#[test]
fn cyrillic_roundtrip() {
    let ru = "Привет, мир! Это тест кириллицы.";
    assert!(roundtrip(ru), "Cyrillic text must round-trip");
}

#[test]
fn long_cyrillic() {
    let ru = "Привет ".repeat(5000);
    assert!(roundtrip(&ru), "long Cyrillic string must round-trip");
}

#[test]
fn mixed_ascii_and_cyrillic() {
    let mixed = "Hello мир world Привет".repeat(100);
    assert!(roundtrip(&mixed));
}

#[test]
fn tab_and_special_whitespace() {
    assert!(roundtrip("\t\t  \t"), "tabs and spaces must round-trip");
}

#[test]
fn encode_is_deterministic() {
    let (enc, _) = make_full_encoder();
    let text = "Hello мир 你好 😀";
    let ids1 = enc.encode(text);
    let ids2 = enc.encode(text);
    assert_eq!(ids1, ids2, "encoding must be deterministic");
}

#[test]
fn decode_unknown_token_returns_error() {
    let trie = Arc::new(VocabTrie::new());
    let dec = Decoder::from_trie(&trie);
    assert!(dec.decode(&[99999]).is_err(), "unknown token id must return Err");
}

#[test]
fn leading_space_flag_roundtrip() {
    // "hello world" with leading-space absorption: " world" token
    let mut trie = VocabTrie::new();
    for b in 0u8..=127 {
        trie.insert(&[b], -8.0);
    }
    trie.insert(b"hello", -1.0);
    trie.insert(b"world", -1.0);
    trie.insert(b" world", -1.5);
    trie.finalize();
    let trie = Arc::new(trie);
    let enc = Encoder::new(Arc::clone(&trie));
    let dec = Decoder::from_trie(&trie);
    let ids = enc.encode("hello world");
    let decoded = dec.decode(&ids).unwrap();
    assert_eq!(decoded, "hello world");
}

#[test]
fn all_printable_ascii_roundtrip() {
    let all_ascii: String = (32u8..=126).map(|b| b as char).collect();
    assert!(roundtrip(&all_ascii), "all printable ASCII must round-trip");
}

#[test]
fn batch_vs_single_consistency() {
    use tokenismo_core::encode_batch;
    let (enc, dec) = make_full_encoder();
    let texts = vec!["Hello", "мир", "😀 test", "", "a"];
    let batch_results = encode_batch(&enc, &texts);
    for (text, batch_ids) in texts.iter().zip(batch_results.iter()) {
        let single_ids = enc.encode(text);
        assert_eq!(&single_ids, batch_ids, "batch must match single encode for {text:?}");
        let decoded = dec.decode(batch_ids).unwrap();
        assert_eq!(*text, decoded, "batch result must round-trip for {text:?}");
    }
}
