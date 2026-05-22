#![no_main]

use libfuzzer_sys::fuzz_target;
use std::sync::{Arc, OnceLock};
use tokenismo_core::{Decoder, Encoder, VocabTrie};

struct FuzzEncoder {
    encoder: Encoder,
    decoder: Decoder,
}

unsafe impl Sync for FuzzEncoder {}

fn get_encoder() -> &'static FuzzEncoder {
    static ENC: OnceLock<FuzzEncoder> = OnceLock::new();
    ENC.get_or_init(|| {
        let mut trie = VocabTrie::new();
        for b in 0u8..=255 {
            trie.insert(&[b], -8.0);
        }
        for word in &[
            "the", " the", "ing", "tion", "er", "in", "on", "an", "at",
            "ен", "ого", "ть", " ", "  ", "\n", "\t",
        ] {
            trie.insert(word.as_bytes(), -2.0);
        }
        let trie = Arc::new(trie);
        FuzzEncoder {
            encoder: Encoder::new(Arc::clone(&trie)),
            decoder: Decoder::from_trie(&trie),
        }
    })
}

fuzz_target!(|data: &[u8]| {
    // Only fuzz valid UTF-8 — invalid byte sequences are rejected at the API boundary.
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    let fe = get_encoder();
    let ids = fe.encoder.encode(text);

    // Invariant 1: encode of empty string is empty
    if text.is_empty() {
        assert!(ids.is_empty(), "empty input must produce no tokens");
        return;
    }

    // Invariant 2: non-empty input produces at least one token
    assert!(!ids.is_empty(), "non-empty input must produce at least one token");

    // Invariant 3: lossless round-trip
    let decoded = fe
        .decoder
        .decode(&ids)
        .expect("decode must not fail on tokens produced by encode");
    assert_eq!(text, decoded, "encode→decode round-trip failed");

    // Invariant 4: determinism
    let ids2 = fe.encoder.encode(text);
    assert_eq!(ids, ids2, "encoding must be deterministic");
});
