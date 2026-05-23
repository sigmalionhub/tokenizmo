use std::collections::HashMap;
use rayon::prelude::*;

/// Switch to Viterbi-based loss when vocab fits in this threshold.
/// Below threshold, HashMap Viterbi is fast (no DARTS construction needed).
const VITERBI_THRESHOLD: usize = 500_000;

const MAX_TOKEN_LEN: usize = 32;

/// Compute approximate loss for each multi-char token.
///
/// Large vocab (> VITERBI_THRESHOLD): fast count proxy
///   loss[tok] = count[tok] * (-log_prob[tok])
///
/// Small vocab (<= VITERBI_THRESHOLD): Viterbi-based actual usage
///   loss[tok] = viterbi_usage[tok] * (-log_prob[tok])
///
/// Viterbi uses a simple HashMap DP — no trie/DARTS construction,
/// so it's fast even for 500k-token vocabs.
pub fn compute_losses(
    multi_tokens: &HashMap<Vec<u8>, f32>,
    counts: &HashMap<Vec<u8>, u64>,
    vocab_log_probs: &HashMap<Vec<u8>, f32>,
    sample_texts: &[String],
) -> HashMap<Vec<u8>, f32> {
    if vocab_log_probs.len() > VITERBI_THRESHOLD {
        // Fast proxy: count * (-log_prob) — prunes rare tokens first.
        return multi_tokens
            .par_iter()
            .map(|(tok, &lp)| {
                let cnt = counts.get(tok).copied().unwrap_or(1) as f32;
                (tok.clone(), cnt * (-lp))
            })
            .collect();
    }

    // Viterbi-based loss: actual encoder usage on sample texts.
    let usage: HashMap<Vec<u8>, u64> = sample_texts
        .par_iter()
        .map(|text| viterbi_usage(vocab_log_probs, text.as_bytes()))
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
            a
        });

    multi_tokens
        .iter()
        .map(|(tok, &lp)| {
            let freq = usage.get(tok).copied().unwrap_or(0) as f32;
            (tok.clone(), freq * (-lp))
        })
        .collect()
}

/// HashMap-based Viterbi DP. No trie — just HashMap lookups.
/// Returns per-token usage counts for this text.
///
/// O(n × MAX_TOKEN_LEN) time, O(n) space.
fn viterbi_usage(vocab_lp: &HashMap<Vec<u8>, f32>, text: &[u8]) -> HashMap<Vec<u8>, u64> {
    let n = text.len();
    if n == 0 {
        return HashMap::new();
    }

    let mut dp_count = vec![u32::MAX; n + 1];
    let mut dp_neg_lp = vec![f32::INFINITY; n + 1];
    let mut dp_prev = vec![0usize; n + 1];
    // Store the starting position of the token that reaches each dp slot.
    let mut dp_start = vec![0usize; n + 1];

    dp_count[0] = 0;
    dp_neg_lp[0] = 0.0;

    for i in 0..n {
        if dp_count[i] == u32::MAX {
            continue;
        }
        let base_count = dp_count[i];
        let base_neg_lp = dp_neg_lp[i];

        let max_end = (i + MAX_TOKEN_LEN).min(n);
        for end in (i + 1)..=max_end {
            let tok = &text[i..end];
            if let Some(&lp) = vocab_lp.get(tok) {
                let new_count = base_count + 1;
                let new_neg_lp = base_neg_lp - lp;
                let better = new_count < dp_count[end]
                    || (new_count == dp_count[end] && new_neg_lp < dp_neg_lp[end]);
                if better {
                    dp_count[end] = new_count;
                    dp_neg_lp[end] = new_neg_lp;
                    dp_prev[end] = i;
                    dp_start[end] = i;
                }
            }
        }
    }

    // Backtrack and accumulate usage.
    let mut usage: HashMap<Vec<u8>, u64> = HashMap::new();
    let mut pos = n;
    while pos > 0 {
        if dp_count[pos] == u32::MAX {
            // Fallback: byte-by-byte (shouldn't happen with well-trained vocab).
            pos -= 1;
            continue;
        }
        let start = dp_start[pos];
        let tok = text[start..pos].to_vec();
        *usage.entry(tok).or_default() += 1;
        pos = dp_prev[pos];
    }
    usage
}
