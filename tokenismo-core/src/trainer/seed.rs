use std::collections::HashMap;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};

/// Returns true if `tok` is a single Unicode scalar value (not byte-length 1).
pub fn is_single_unicode(tok: &[u8]) -> bool {
    match std::str::from_utf8(tok) {
        Ok(s) => s.chars().count() == 1,
        Err(_) => false,
    }
}

/// Extract all substrings up to `max_token_len` bytes from each document,
/// counting frequencies. Runs in parallel via Rayon.
///
/// Only byte sequences starting at a UTF-8 character boundary are considered
/// (continuation bytes 0x80–0xBF are skipped as start positions).
pub fn seed_candidates(
    texts: &[String],
    max_token_len: usize,
    min_freq: usize,
) -> HashMap<Vec<u8>, u64> {
    const BATCH_SIZE: usize = 2_000;

    let n_batches = texts.len().div_ceil(BATCH_SIZE);
    let pb = ProgressBar::new(n_batches as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "  Seeding  [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} batches  ({eta} left)",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let chunks: Vec<&[String]> = texts.chunks(BATCH_SIZE).collect();
    let mut merged = chunks
        .into_par_iter()
        .progress_with(pb.clone())
        .map(|batch| {
            let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
            let mut starts_buf: Vec<usize> = Vec::new();
            for text in batch {
                count_text_substrings(text.as_bytes(), max_token_len, &mut counts, &mut starts_buf);
            }
            counts.retain(|tok, &mut cnt| cnt >= min_freq as u64 || is_single_unicode(tok));
            counts
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_default() += v;
            }
            a
        });

    pb.finish_and_clear();

    // Global prune on merged totals (per-batch pruning is approximate).
    merged.retain(|tok, &mut cnt| cnt >= min_freq as u64 || is_single_unicode(tok));
    merged
}

fn count_text_substrings(
    data: &[u8],
    max_token_len: usize,
    counts: &mut HashMap<Vec<u8>, u64>,
    starts_buf: &mut Vec<usize>,
) {
    // Split on newlines, then apply the same word-unit pre-tokenization as the
    // encoder's `split_individual_words`:
    //   - multi-space (≥2) → isolated unit (the spaces ARE the token bytes)
    //   - single leading space → stripped (becomes LEADING_SPACE_FLAG at encode time)
    //   - no leading space → count as-is
    // This ensures indentation blocks like "    " become high-frequency candidates.
    for raw_line in data.split(|&b| b == b'\n') {
        let line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
        for unit in split_line_into_units(line) {
            // Strip single leading space — it's LEADING_SPACE_FLAG, not a vocab byte.
            let count_bytes = if unit.len() >= 2 && unit[0] == b' ' && unit[1] != b' ' {
                &unit[1..]
            } else {
                unit
            };
            if count_bytes.is_empty() {
                continue;
            }
            let n = count_bytes.len();
            starts_buf.clear();
            starts_buf.extend((0..n).filter(|&i| !(0x80..=0xBF).contains(&count_bytes[i])));
            let ns = starts_buf.len();
            for si in 0..ns {
                let s = starts_buf[si];
                for ei in (si + 1)..=(ns) {
                    let e = if ei < ns { starts_buf[ei] } else { n };
                    if e - s > max_token_len {
                        break;
                    }
                    *counts.entry(count_bytes[s..e].to_vec()).or_default() += 1;
                }
            }
        }
    }
}

/// Split one line of text into word units using the same logic as the encoder's
/// `split_individual_words`: multi-space (≥2) → own unit; single space →
/// stays attached to the following word.
fn split_line_into_units(line: &[u8]) -> Vec<&[u8]> {
    let n = line.len();
    let mut units = Vec::new();
    let mut i = 0;
    while i < n {
        let start = i;
        while i < n && line[i] == b' ' { i += 1; }
        let space_count = i - start;
        if space_count > 1 {
            units.push(&line[start..i]);
            let word_start = i;
            while i < n && line[i] != b' ' { i += 1; }
            if i > word_start { units.push(&line[word_start..i]); }
        } else {
            while i < n && line[i] != b' ' { i += 1; }
            if i > start { units.push(&line[start..i]); }
        }
    }
    units
}

/// Add guaranteed single-character tokens so the encoder never fails.
/// Covers ASCII, Cyrillic, Latin Extended, Greek, and a CJK sample.
pub fn add_guaranteed_tokens(counts: &mut HashMap<Vec<u8>, u64>) {
    // All 128 ASCII bytes.
    for byte_val in 0u8..=127 {
        counts.entry(vec![byte_val]).or_insert(1);
    }

    // Unicode ranges: insert each char's UTF-8 bytes with count=1 if absent.
    const RANGES: &[(u32, u32)] = &[
        (0x0080, 0x0250), // Latin Extended
        (0x0370, 0x0400), // Greek and Coptic
        (0x0400, 0x0530), // Cyrillic + Supplement
        (0x4E00, 0x4F00), // CJK sample (first 256)
    ];
    for &(start, end) in RANGES {
        for cp in start..end {
            if let Some(ch) = char::from_u32(cp) {
                let mut buf = [0u8; 4];
                let encoded = ch.encode_utf8(&mut buf);
                counts.entry(encoded.as_bytes().to_vec()).or_insert(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_substrings() {
        let texts = vec!["ab ab".to_string()];
        let counts = seed_candidates(&texts, 4, 1);
        assert_eq!(*counts.get(b"ab".as_slice()).unwrap_or(&0), 2);
        assert_eq!(*counts.get(b"a".as_slice()).unwrap_or(&0), 2);
    }

    #[test]
    fn guaranteed_ascii_present() {
        let mut counts = HashMap::new();
        add_guaranteed_tokens(&mut counts);
        for b in 0u8..=127 {
            assert!(counts.contains_key(&vec![b]), "missing ASCII byte {b}");
        }
    }

    #[test]
    fn respects_utf8_boundaries() {
        // "привет" is 12 bytes (2 bytes per Cyrillic char in UTF-8)
        let texts = vec!["привет".to_string()];
        let counts = seed_candidates(&texts, 32, 1);
        // The full word should be a candidate.
        let full = "привет".as_bytes().to_vec();
        assert!(counts.contains_key(&full));
        // A mid-character slice like bytes 1..3 should NOT be a candidate
        // (starts at continuation byte 0xBF).
        let bad = "привет".as_bytes()[1..3].to_vec();
        assert!(!counts.contains_key(&bad), "continuation-byte slice should not be a candidate");
    }
}
