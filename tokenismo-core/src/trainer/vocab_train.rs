use std::collections::HashMap;

pub struct TrainEntry {
    pub token: Vec<u8>,
    pub log_prob: f32,
}

/// Ordered vocabulary used during training.
/// Token IDs are assigned in order of descending log_prob (highest prob = ID 0).
pub struct TrainVocab {
    pub entries: Vec<TrainEntry>,
    token_to_id: HashMap<Vec<u8>, usize>,
}

impl TrainVocab {
    pub fn from_log_probs(map: HashMap<Vec<u8>, f32>) -> Self {
        let mut pairs: Vec<(Vec<u8>, f32)> = map.into_iter().collect();
        // Sort descending by log_prob; ties broken by token bytes (for determinism).
        pairs.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let mut entries = Vec::with_capacity(pairs.len());
        let mut token_to_id = HashMap::with_capacity(pairs.len());
        for (token, log_prob) in pairs {
            let id = entries.len();
            token_to_id.insert(token.clone(), id);
            entries.push(TrainEntry { token, log_prob });
        }
        Self { entries, token_to_id }
    }

    pub fn get_id(&self, token: &[u8]) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &TrainEntry> {
        self.entries.iter()
    }
}
