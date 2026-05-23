pub mod normalizer;
pub mod corpus;
pub mod vocab_train;
pub mod seed;
pub mod loss;
pub mod em;

pub use normalizer::normalize;
pub use corpus::{load_corpus_config, collect_documents};
pub use vocab_train::TrainVocab;
pub use em::train;
