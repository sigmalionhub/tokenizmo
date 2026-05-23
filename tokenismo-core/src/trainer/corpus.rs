use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::fs::File;
use flate2::read::GzDecoder;
use serde::Deserialize;
use anyhow::{Context, Result};

#[derive(Debug, Deserialize)]
struct ConfigFile {
    mix_ratios: std::collections::HashMap<String, f32>,
    sources: Vec<SourceConfig>,
}

#[derive(Debug, Deserialize)]
struct SourceConfig {
    path: String,
    language: String,
    #[serde(default = "default_reader")]
    reader: String,
    #[serde(default = "default_text_field")]
    text_field: String,
}

fn default_reader() -> String { "text".into() }
fn default_text_field() -> String { "text".into() }

pub struct CorpusSource {
    pub path: PathBuf,
    pub language: String,
    pub weight: f32,
    pub reader: String,
    pub text_field: String,
}

/// Load corpus config from a YAML file and return weighted sources.
/// `base_dir` is the project root (paths in config are relative to it).
pub fn load_corpus_config(config_path: &Path, base_dir: &Path) -> Result<Vec<CorpusSource>> {
    let content = std::fs::read_to_string(config_path)
        .with_context(|| format!("reading config {}", config_path.display()))?;
    let cfg: ConfigFile = serde_yaml::from_str(&content)
        .with_context(|| format!("parsing config {}", config_path.display()))?;

    let sources = cfg.sources.into_iter().map(|s| {
        let weight = cfg.mix_ratios.get(&s.language).copied().unwrap_or(1.0);
        CorpusSource {
            path: base_dir.join(&s.path),
            language: s.language,
            weight,
            reader: s.reader,
            text_field: s.text_field,
        }
    }).collect();

    Ok(sources)
}

/// Collect all documents from a list of sources into memory, weighted by their `weight` field.
/// Uses simple weighted interleaving: all source files are read up front then merged.
/// For large corpora, weights control how many docs each source contributes.
/// Note: the entire corpus is held in memory; callers must size `max_docs` accordingly.
pub fn collect_documents(sources: &[CorpusSource], max_docs: usize) -> Result<Vec<String>> {
    if sources.is_empty() {
        return Ok(Vec::new());
    }

    let total_weight: f32 = sources.iter().map(|s| s.weight).sum();
    let mut all_docs: Vec<String> = Vec::new();

    // Calculate target document counts per source based on weights.
    // We collect all docs then sample proportionally.
    let mut per_source: Vec<Vec<String>> = Vec::with_capacity(sources.len());
    for src in sources {
        let docs = read_source(src)?;
        per_source.push(docs);
    }

    // Interleave sources proportionally by weight.
    // Use a simple round-robin with weight-based token accumulation.
    let mut tokens: Vec<f32> = sources.iter().map(|s| s.weight / total_weight).collect();
    let mut indices: Vec<usize> = vec![0; sources.len()];
    let all_exhausted = |idx: &[usize], docs: &[Vec<String>]| {
        idx.iter().zip(docs.iter()).all(|(&i, d)| i >= d.len())
    };

    while !all_exhausted(&indices, &per_source) {
        if max_docs > 0 && all_docs.len() >= max_docs {
            break;
        }
        // Pick source with highest accumulated token.
        let chosen = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| indices[*i] < per_source[*i].len())
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);

        if let Some(i) = chosen {
            let doc = per_source[i][indices[i]].clone();
            if !doc.trim().is_empty() {
                all_docs.push(doc);
            }
            indices[i] += 1;
            tokens[i] -= 1.0;
            // Replenish all tokens each round.
            for (j, src) in sources.iter().enumerate() {
                tokens[j] += src.weight / total_weight;
            }
        } else {
            break;
        }
    }

    Ok(all_docs)
}

fn read_source(src: &CorpusSource) -> Result<Vec<String>> {
    if !src.path.exists() {
        eprintln!("  WARNING: corpus file not found: {}", src.path.display());
        return Ok(Vec::new());
    }
    match src.reader.as_str() {
        "text" => read_text(&src.path),
        "text_gz" => read_text_gz(&src.path),
        "jsonl" => read_jsonl(&src.path, &src.text_field),
        "jsonl_gz" => read_jsonl_gz(&src.path, &src.text_field),
        other => anyhow::bail!("unknown reader: {}", other),
    }
}

fn read_text(path: &Path) -> Result<Vec<String>> {
    let f = File::open(path)?;
    collect_paragraphs(BufReader::new(f))
}

fn read_text_gz(path: &Path) -> Result<Vec<String>> {
    let f = File::open(path)?;
    collect_paragraphs(BufReader::new(GzDecoder::new(f)))
}

fn collect_paragraphs<R: BufRead>(reader: R) -> Result<Vec<String>> {
    let mut docs = Vec::new();
    let mut buf: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            if !buf.is_empty() {
                docs.push(buf.join("\n"));
                buf.clear();
            }
        } else {
            buf.push(line);
        }
    }
    if !buf.is_empty() {
        docs.push(buf.join("\n"));
    }
    Ok(docs)
}

fn read_jsonl(path: &Path, text_field: &str) -> Result<Vec<String>> {
    let f = File::open(path)?;
    collect_jsonl(BufReader::new(f), text_field)
}

fn read_jsonl_gz(path: &Path, text_field: &str) -> Result<Vec<String>> {
    let f = File::open(path)?;
    collect_jsonl(BufReader::new(GzDecoder::new(f)), text_field)
}

fn collect_jsonl<R: BufRead>(reader: R, text_field: &str) -> Result<Vec<String>> {
    let mut docs = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(obj) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = obj.get(text_field).and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    docs.push(text.to_string());
                }
            }
        }
    }
    Ok(docs)
}
