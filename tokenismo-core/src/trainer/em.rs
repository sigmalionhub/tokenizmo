use std::collections::HashMap;
use indicatif::{ProgressBar, ProgressStyle};
use super::seed::{seed_candidates, add_guaranteed_tokens, is_single_unicode};
use super::loss::compute_losses;
use super::vocab_train::TrainVocab;

// Special tokens always assigned first (IDs 0–3) with log_prob = 0.0.
const SPECIAL_TOKENS: &[&[u8]] = &[b"<unk>", b"<s>", b"</s>", b"<pad>"];

fn estimate_log_probs(counts: &HashMap<Vec<u8>, u64>) -> HashMap<Vec<u8>, f32> {
    let total: u64 = counts.values().sum();
    if total == 0 {
        return HashMap::new();
    }
    let log_total = (total as f64).ln();
    counts
        .iter()
        .filter(|(_, &c)| c > 0)
        .map(|(tok, &cnt)| {
            let lp = (cnt as f64).ln() - log_total;
            (tok.clone(), lp as f32)
        })
        .collect()
}

pub fn train(
    texts: &[String],
    vocab_size: usize,
    max_token_len: usize,
    shrink_factor: f32,
    min_freq: usize,
) -> TrainVocab {
    println!("  Seeding candidates from {} documents...", texts.len());
    let mut counts = seed_candidates(texts, max_token_len, min_freq);
    add_guaranteed_tokens(&mut counts);

    // Final prune after guaranteed tokens are added.
    counts.retain(|tok, &mut cnt| cnt >= min_freq as u64 || is_single_unicode(tok));

    println!("  Initial candidates: {}", counts.len());

    let log_probs = estimate_log_probs(&counts);

    // Separate single Unicode characters (never pruned) from multi-char tokens.
    let mut single_chars: HashMap<Vec<u8>, f32> = HashMap::new();
    let mut multi_tokens: HashMap<Vec<u8>, f32> = HashMap::new();
    for (tok, lp) in log_probs {
        if is_single_unicode(&tok) {
            single_chars.insert(tok, lp);
        } else {
            multi_tokens.insert(tok, lp);
        }
    }

    let special_count = SPECIAL_TOKENS.len();
    let target_multi = vocab_size.saturating_sub(single_chars.len() + special_count);

    let mut iteration = 0usize;
    let em_sample_size = 2000.min(texts.len());

    // Estimate iterations: each shrinks by shrink_factor until target_multi reached.
    let est_iters = if multi_tokens.len() > target_multi && target_multi > 0 {
        ((multi_tokens.len() as f32 / target_multi as f32).ln()
            / (1.0 / shrink_factor).ln())
        .ceil() as u64
    } else {
        1
    };

    let pb = ProgressBar::new(est_iters);
    pb.set_style(
        ProgressStyle::with_template(
            "  EM  [{elapsed_precise}] [{bar:40.green/black}] iter {pos}/{len}  {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    while multi_tokens.len() > target_multi {
        iteration += 1;
        let keep_n = target_multi.max((multi_tokens.len() as f32 * shrink_factor) as usize);

        // Build combined vocab for Viterbi iterations (single_chars + multi_tokens).
        let vocab_lp: HashMap<Vec<u8>, f32> = single_chars
            .iter()
            .chain(multi_tokens.iter())
            .map(|(k, &v)| (k.clone(), v))
            .collect();

        let losses = compute_losses(&multi_tokens, &counts, &vocab_lp, &texts[..em_sample_size]);

        // Keep tokens with highest loss (removing them hurts most).
        let mut sorted: Vec<(Vec<u8>, f32)> = multi_tokens.into_iter().collect();
        sorted.sort_unstable_by(|(ta, _), (tb, _)| {
            let la = losses.get(ta).copied().unwrap_or(0.0);
            let lb = losses.get(tb).copied().unwrap_or(0.0);
            lb.partial_cmp(&la).unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(keep_n);

        // Re-estimate log_probs on remaining vocabulary.
        let remaining_counts: HashMap<Vec<u8>, u64> = single_chars
            .keys()
            .chain(sorted.iter().map(|(t, _)| t))
            .filter_map(|tok| counts.get(tok).map(|&c| (tok.clone(), c)))
            .collect();

        let new_lp = estimate_log_probs(&remaining_counts);

        // Update single_chars log_probs.
        for (tok, lp) in &mut single_chars {
            if let Some(&new) = new_lp.get(tok) {
                *lp = new;
            }
        }

        multi_tokens = sorted
            .into_iter()
            .map(|(tok, lp)| {
                let new = new_lp.get(&tok).copied().unwrap_or(lp);
                (tok, new)
            })
            .collect();

        let total = single_chars.len() + multi_tokens.len() + special_count;
        pb.set_message(format!("{total} tokens → target {vocab_size}"));
        pb.inc(1);

        if multi_tokens.is_empty() {
            break;
        }
    }
    pb.finish_and_clear();
    println!("  EM complete: {} iterations", iteration);

    // Build final vocabulary: special tokens first (lp=0.0), then rest by lp desc.
    let mut final_lp: HashMap<Vec<u8>, f32> =
        HashMap::with_capacity(special_count + single_chars.len() + multi_tokens.len());

    for &tok in SPECIAL_TOKENS {
        final_lp.insert(tok.to_vec(), 0.0);
    }
    for (tok, lp) in single_chars {
        final_lp.entry(tok).or_insert(lp);
    }
    for (tok, lp) in multi_tokens {
        final_lp.entry(tok).or_insert(lp);
    }

    TrainVocab::from_log_probs(final_lp)
}
