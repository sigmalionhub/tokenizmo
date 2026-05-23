//! TokeNismo core tokenizer library.
//!
//! Provides a Unigram Language Model tokenizer with Viterbi DP segmentation.
//!
//! # Quick start
//!
//! ```no_run
//! use std::sync::Arc;
//! use tokenismo_core::{Encoder, Decoder, VocabTrie};
//! use tokenismo_core::vocab_io::load_vocab;
//!
//! let trie = Arc::new(load_vocab("my.vocab".as_ref()).unwrap());
//! let encoder = Encoder::new(Arc::clone(&trie));
//! let decoder = Decoder::from_trie(&trie);
//!
//! let ids = encoder.encode("Hello world");
//! let text = decoder.decode(&ids).unwrap();
//! assert_eq!(text, "Hello world");
//! ```

pub mod vocab;
pub mod encoder;
pub mod decoder;
pub mod batch;
pub mod vocab_io;
pub mod trainer;

pub use encoder::Encoder;
pub use decoder::Decoder;
pub use vocab::VocabTrie;
pub use batch::encode_batch;
