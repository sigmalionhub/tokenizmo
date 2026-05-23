use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokenismo_core::{vocab_io, Encoder};

fn bench(label: &str, enc: &Encoder, text: &str, iters: u32) {
    // warmup
    for _ in 0..10 { let _ = enc.encode(text); }

    let start = Instant::now();
    for _ in 0..iters { let _ = enc.encode(text); }
    let elapsed = start.elapsed();

    let bytes_per_iter = text.len() as f64;
    let total_bytes = bytes_per_iter * iters as f64;
    let mb_s = total_bytes / elapsed.as_secs_f64() / 1_048_576.0;
    println!("{label:30} {:>7} bytes  {:>8.1} MiB/s  ({iters} iters, {:.1}ms total)",
        text.len(), mb_s, elapsed.as_secs_f64() * 1000.0);
}

fn main() {
    let candidates = [
        "data/vocab/tokenismo_262k_rust_v4.vocab",
        "data/vocab/tokenismo_262k_rust_v3.vocab",
        "data/vocab/tokenismo.vocab",
    ];
    let Some(path) = candidates.iter().find(|p| Path::new(p).exists()).copied() else {
        eprintln!("No vocab file found in data/vocab/");
        return;
    };

    eprintln!("Loading vocab: {path}");
    let trie = Arc::new(vocab_io::load_vocab(Path::new(path)).expect("load vocab"));
    let enc = Encoder::new(Arc::clone(&trie));
    eprintln!("Vocab size: {}", trie.vocab_size);
    println!();

    let en_word  = "The tokenizer converts raw text into token IDs. ";
    let ru_word  = "Токенизатор преобразует исходный текст в токены. ";
    let code_word = "fn encode(text: &str) -> Vec<u32> { todo!() }\n";

    // Input sizes: 256B, 1KB, 4KB, 16KB, 64KB, 256KB
    for &size_kb in &[0u64, 1, 4, 16, 64, 256] {
        let size_b = if size_kb == 0 { 256 } else { size_kb as usize * 1024 };
        let iters = (5_000_000 / size_b).max(5) as u32;

        let en   = en_word.repeat((size_b / en_word.len()).max(1));
        let ru   = ru_word.repeat((size_b / ru_word.len()).max(1));
        let code = code_word.repeat((size_b / code_word.len()).max(1));

        let label = if size_kb == 0 { "256B".to_string() } else { format!("{}KB", size_kb) };
        bench(&format!("EN  {label}"), &enc, &en[..en.len().min(size_b)], iters);
        bench(&format!("RU  {label}"), &enc, &ru[..ru.len().min(size_b)], iters);
        bench(&format!("code {label}"), &enc, &code[..code.len().min(size_b)], iters);
        println!();
    }
}
