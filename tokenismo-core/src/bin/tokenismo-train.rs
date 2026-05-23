use std::path::PathBuf;
use clap::Parser;
use anyhow::{Context, Result};
use tokenismo_core::trainer::{normalize, load_corpus_config, collect_documents, train};
use tokenismo_core::vocab_io::save_vocab_from_train;

#[derive(Parser)]
#[command(name = "tokenismo-train", about = "Train a TokeNismo vocabulary")]
struct Args {
    /// YAML corpus config file
    #[arg(long)]
    config: PathBuf,

    /// Target vocabulary size
    #[arg(long, default_value = "262144")]
    vocab_size: usize,

    /// Output .vocab binary file
    #[arg(long)]
    output: PathBuf,

    /// EM shrink factor (fraction of multi-char tokens to keep per iteration)
    #[arg(long, default_value = "0.75")]
    shrink_factor: f32,

    /// Maximum token byte length
    #[arg(long, default_value = "32")]
    max_token_len: usize,

    /// Minimum corpus frequency for a candidate token
    #[arg(long, default_value = "2")]
    min_freq: usize,

    /// Limit documents loaded (0 = no limit)
    #[arg(long, default_value = "0")]
    max_docs: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let t_start = std::time::Instant::now();
    println!("{}", "=".repeat(60));
    println!("TokeNismo Vocabulary Trainer (Rust)");
    println!("  vocab_size   = {}", args.vocab_size);
    println!("  output       = {}", args.output.display());
    println!("  config       = {}", args.config.display());
    println!("  shrink_factor= {}", args.shrink_factor);
    println!("  max_token_len= {}", args.max_token_len);
    println!("  min_freq     = {}", args.min_freq);
    println!("{}", "=".repeat(60));

    // Determine project root: two levels up from the binary (target/release → project root).
    // Fallback: use current working directory.
    let base_dir = std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."));

    println!("\nLoading corpus config...");
    let sources = load_corpus_config(&args.config, &base_dir)
        .with_context(|| format!("loading config {}", args.config.display()))?;
    println!("  Corpus sources: {}", sources.len());
    for src in &sources {
        println!("    [{:.0}%] {} ({})", src.weight * 100.0, src.path.display(), src.language);
    }

    println!("\nCollecting documents...");
    let raw_docs = collect_documents(&sources, args.max_docs)
        .context("reading corpus")?;
    println!("  Raw documents: {}", raw_docs.len());

    println!("Normalizing...");
    let docs: Vec<String> = raw_docs.iter().map(|d| normalize(d)).collect();
    let total_chars: usize = docs.iter().map(|d| d.chars().count()).sum();
    println!(
        "  Collected {} documents, {} total chars",
        docs.len(),
        total_chars
    );

    if docs.is_empty() {
        anyhow::bail!("No documents loaded. Check corpus config and file paths.");
    }

    println!("\nTraining vocabulary (target size: {})...", args.vocab_size);
    let vocab = train(
        &docs,
        args.vocab_size,
        args.max_token_len,
        args.shrink_factor,
        args.min_freq,
    );
    println!("  Final vocabulary: {} tokens", vocab.len());

    println!("\nValidating vocabulary...");
    validate_vocab(&vocab, &docs);

    println!("\nSaving vocabulary...");
    save_vocab_from_train(&vocab, &args.output)
        .with_context(|| format!("saving vocab to {}", args.output.display()))?;
    println!("  Saved {} tokens to {}", vocab.len(), args.output.display());

    println!("\nDone in {:.1}s", t_start.elapsed().as_secs_f64());
    Ok(())
}

fn validate_vocab(vocab: &tokenismo_core::trainer::TrainVocab, docs: &[String]) {
    use tokenismo_core::VocabTrie;

    // Build a lightweight trie (no DARTS) just for ASCII lookup.
    let mut trie = VocabTrie::new();
    for entry in vocab.iter() {
        trie.insert(&entry.token, entry.log_prob);
    }

    // Check all printable ASCII present (CSR lookup, no finalize needed).
    let missing_ascii: Vec<u8> = (32u8..127)
        .filter(|&b| trie.get(&[b]).is_none())
        .collect();
    if missing_ascii.is_empty() {
        println!("  [OK] All printable ASCII chars present");
    } else {
        println!(
            "  WARNING: {} ASCII chars missing: {:?}",
            missing_ascii.len(),
            &missing_ascii[..missing_ascii.len().min(20)]
        );
    }

    // For large vocabs skip Viterbi round-trip (DARTS build takes too long).
    // Report token length stats instead.
    let total_tokens = vocab.len();
    let avg_token_bytes: f64 = vocab.iter().map(|e| e.token.len() as f64).sum::<f64>()
        / total_tokens as f64;
    let multi_char = vocab.iter().filter(|e| e.token.len() > 1).count();
    println!("  Avg token byte length: {:.2}", avg_token_bytes);
    println!("  Multi-byte tokens: {} ({:.1}%)", multi_char, 100.0 * multi_char as f64 / total_tokens as f64);

    // Chars/token estimate on first doc using actual Viterbi DP (matches encoder behaviour).
    if let Some(doc) = docs.first() {
        let end = doc.char_indices().nth(500).map(|(i, _)| i).unwrap_or(doc.len());
        let sample: &str = &doc[..end];
        let total_chars = sample.chars().count();

        trie.finalize();
        let enc = tokenismo_core::Encoder::new(std::sync::Arc::new(trie));
        let token_count = enc.encode(sample).len();
        if token_count > 0 {
            println!("  Chars/token (Viterbi, 1st doc 500 chars): {:.2}",
                total_chars as f64 / token_count as f64);
        }
    }
}
