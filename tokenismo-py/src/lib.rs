use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use tokenismo_core::{Encoder, Decoder};
use tokenismo_core::vocab_io::load_vocab;
use std::sync::Arc;
use std::path::Path;

#[pyclass]
pub struct TokeNismo {
    encoder: Encoder,
    decoder: Decoder,
}

#[pymethods]
impl TokeNismo {
    /// Load a tokenizer from a .vocab binary file.
    #[staticmethod]
    pub fn from_file(vocab_path: &str) -> PyResult<Self> {
        let trie = load_vocab(Path::new(vocab_path))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let arc = Arc::new(trie);
        Ok(Self {
            decoder: Decoder::from_trie(&arc),
            encoder: Encoder::new(arc),
        })
    }

    /// Encode a string into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encoder.encode(text)
    }

    /// Encode a batch of strings in parallel. Releases the GIL.
    pub fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<u32>> {
        py.allow_threads(|| {
            let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
            tokenismo_core::encode_batch(&self.encoder, &refs)
        })
    }

    /// Decode token IDs back to a string.
    pub fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.decoder.decode(&ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode a batch of token ID sequences. Releases the GIL.
    pub fn decode_batch(&self, py: Python<'_>, ids_batch: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        py.allow_threads(|| {
            ids_batch.iter()
                .map(|ids| self.decoder.decode(ids))
                .collect::<Result<Vec<_>, _>>()
        }).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.encoder.trie.vocab_size
    }
}

#[pymodule]
fn tokenismo(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TokeNismo>()?;
    Ok(())
}
