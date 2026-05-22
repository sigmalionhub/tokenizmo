use std::sync::{Arc, OnceLock};
use tokenismo_core::{Decoder, Encoder, VocabTrie};
use proptest::prelude::*;

struct TestEncoder {
    encoder: Encoder,
    decoder: Decoder,
}

unsafe impl Sync for TestEncoder {}

fn shared_encoder() -> &'static TestEncoder {
    static INST: OnceLock<TestEncoder> = OnceLock::new();
    INST.get_or_init(|| {
        let mut trie = VocabTrie::new();
        // All 256 single-byte tokens guarantee full coverage of any byte sequence.
        for b in 0u8..=255 {
            trie.insert(&[b], -8.0);
        }
        // Common multi-byte tokens improve compression and exercise the trie walk.
        for word in &[
            "the", " the", "ing", "tion", "er", "re", "in", "on", "an", "at",
            "ен", "ого", "ии", "ть", "ние",
        ] {
            trie.insert(word.as_bytes(), -2.0);
        }
        trie.finalize();
        let trie = Arc::new(trie);
        TestEncoder {
            encoder: Encoder::new(Arc::clone(&trie)),
            decoder: Decoder::from_trie(&trie),
        }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn roundtrip(s in any::<String>()) {
        let te = shared_encoder();
        let ids = te.encoder.encode(&s);
        let decoded = te.decoder.decode(&ids)
            .expect("decode must not fail on tokens produced by encode");
        prop_assert_eq!(s, decoded, "encode→decode must be lossless");
    }

    #[test]
    fn encode_is_deterministic(s in any::<String>()) {
        let te = shared_encoder();
        let ids1 = te.encoder.encode(&s);
        let ids2 = te.encoder.encode(&s);
        prop_assert_eq!(ids1, ids2, "two encodes of the same string must be identical");
    }

    #[test]
    fn token_count_positive_for_nonempty(s in ".+") {
        let te = shared_encoder();
        let ids = te.encoder.encode(&s);
        prop_assert!(!ids.is_empty(), "non-empty string must produce at least one token");
    }

    #[test]
    fn empty_string_always_empty(_s in Just("".to_string())) {
        let te = shared_encoder();
        let ids = te.encoder.encode("");
        prop_assert_eq!(ids, vec![], "empty string must produce no tokens");
    }
}
