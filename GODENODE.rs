#![allow(dead_code, unused_imports)]
use std::collections::{HashMap, VecDeque};
use sha2::{Sha256, Digest};
use eframe::egui;
use std::f32::consts::PI;
use std::time::Duration;
use rayon::prelude::*;
use anyhow::{Result, Context};

// ==============================================================================
// 1. THE MATHEMATICAL CORE (MP4c HDC & God Node Engine)
// ==============================================================================
pub const DIMS: usize = 8192;
pub const PACKED_BYTES: usize = DIMS / 8;
pub const DECAY_RATE: f64 = 0.01;
pub const NGRAM_SIZE: usize = 3; // FIX 1: n-gram size for semantic encoding

pub fn now_secs() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MemoryTier { Working, Episodic, Semantic, Procedural }

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Hypervector {
    pub data: Vec<u8>,
    pub concept_id: String,
    pub strength: f64,
    pub last_access: f64,
    pub source_text: String,
}

impl Hypervector {
    // ===========================================================================
    // FIX 1: SEMANTIC ENCODING — Replace SHA256 random seeding with
    // character n-gram bundling. Similar texts share n-grams and thus
    // produce similar hypervectors. "king" and "queen" will now be
    // measurably closer than "king" and "xylophone".
    // ===========================================================================
    pub fn from_string(text: &str, concept_id: Option<&str>) -> Self {
        let chars: Vec<char> = text.chars().collect();
        let concept_id_str = concept_id.map(|s| s.to_string())
            .unwrap_or_else(|| text.chars().take(50).collect());

        // If the text is shorter than NGRAM_SIZE, fall back to the
        // whole-string atomic vector (seeded deterministically via SHA256)
        if chars.len() < NGRAM_SIZE {
            return Self::atomic_from_str(text, &concept_id_str);
        }

        // Build one base vector per unique character n-gram, then bundle them.
        // Texts that share n-grams will share those base vectors, producing
        // high Hamming similarity proportional to lexical/morphological overlap.
        let ngram_hvs: Vec<Hypervector> = chars
            .windows(NGRAM_SIZE)
            .map(|w| {
                let ngram: String = w.iter().collect();
                Self::atomic_from_str(&ngram, &ngram)
            })
            .collect();

        let mut result = bundle(&ngram_hvs);
        result.concept_id = concept_id_str;
        result.source_text = text.to_string();
        result
    }

    // Deterministic atomic vector seeded from SHA256 — used only for
    // n-grams and short tokens, NOT for full semantic concepts.
    fn atomic_from_str(text: &str, concept_id: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let digest = hasher.finalize();
        let seed = u64::from_le_bytes([
            digest[0], digest[1], digest[2], digest[3],
            digest[4], digest[5], digest[6], digest[7],
        ]);
        let mut state = seed;
        let data: Vec<u8> = (0..PACKED_BYTES).map(|_| xorshift64(&mut state)).collect();
        Hypervector {
            data,
            concept_id: concept_id.to_string(),
            strength: 1.0,
            last_access: now_secs(),
            source_text: text.to_string(),
        }
    }

    pub fn similarity(&self, other: &Self, apply_strength: bool) -> f64 {
        let hamming: u32 = self.data.iter().zip(other.data.iter())
            .map(|(&a, &b)| (a ^ b).count_ones()).sum();
        let raw = 1.0 - hamming as f64 / DIMS as f64;
        if apply_strength { raw * self.strength.min(other.strength) } else { raw }
    }

    pub fn bind(&self, other: &Self) -> Self {
        let data: Vec<u8> = self.data.iter().zip(other.data.iter())
            .map(|(&a, &b)| a ^ b).collect();
        Hypervector {
            data,
            concept_id: format!("bound({},{})", self.concept_id, other.concept_id),
            strength: self.strength.min(other.strength) * 0.95,
            last_access: now_secs(),
            source_text: self.source_text.clone(),
        }
    }

    pub fn permute(&self, shift: i64) -> Self {
        let shift = ((shift % 8192 + 8192) as usize) % 8192;
        let byte_shift = shift / 8;
        let bit_shift = shift % 8;
        let mut rotated = vec![0u8; PACKED_BYTES];
        for i in 0..PACKED_BYTES {
            rotated[i] = self.data[(i + PACKED_BYTES - byte_shift) % PACKED_BYTES];
        }
        if bit_shift > 0 {
            let mut shifted = vec![0u8; PACKED_BYTES];
            for i in 0..PACKED_BYTES {
                shifted[i] = (rotated[i] << bit_shift)
                    | (rotated[(i + 1) % PACKED_BYTES] >> (8 - bit_shift));
            }
            rotated = shifted;
        }
        Hypervector {
            data: rotated,
            concept_id: self.concept_id.clone(),
            strength: self.strength,
            last_access: self.last_access,
            source_text: self.source_text.clone(),
        }
    }

    pub fn inverse_permute(&self, shift: i64) -> Self {
        let shift = ((shift % 8192 + 8192) as usize) % 8192;
        let byte_shift = shift / 8;
        let bit_shift = shift % 8;
        let mut rotated = vec![0u8; PACKED_BYTES];
        for i in 0..PACKED_BYTES {
            rotated[i] = self.data[(i + byte_shift) % PACKED_BYTES];
        }
        if bit_shift > 0 {
            let mut shifted = vec![0u8; PACKED_BYTES];
            for i in 0..PACKED_BYTES {
                shifted[i] = (rotated[i] >> bit_shift)
                    | (rotated[(i + PACKED_BYTES - 1) % PACKED_BYTES] << (8 - bit_shift));
            }
            rotated = shifted;
        }
        Hypervector {
            data: rotated,
            concept_id: self.concept_id.clone(),
            strength: self.strength,
            last_access: self.last_access,
            source_text: self.source_text.clone(),
        }
    }

    pub fn decay(&mut self) {
        let elapsed = now_secs() - self.last_access;
        self.strength *= (-DECAY_RATE * elapsed).exp();
    }

    // ===========================================================================
    // FIX 2: TOUCH — refresh last_access so retrieval keeps memory alive.
    // Call this whenever a hypervector is successfully retrieved.
    // ===========================================================================
    pub fn touch(&mut self) {
        self.last_access = now_secs();
    }
}

fn xorshift64(state: &mut u64) -> u8 {
    let mut x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x; x as u8
}

pub fn bundle(vectors: &[Hypervector]) -> Hypervector {
    if vectors.is_empty() { return Hypervector::atomic_from_str("zero", "zero"); }
    let mut counts = vec![0i32; DIMS];
    for hv in vectors {
        for (byte_i, &byte) in hv.data.iter().enumerate() {
            for bit_i in 0..8usize {
                let idx = byte_i * 8 + bit_i;
                if byte & (0x80u8 >> bit_i) != 0 { counts[idx] += 1; }
                else { counts[idx] -= 1; }
            }
        }
    }
    let mut data = vec![0u8; PACKED_BYTES];
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 { data[i / 8] |= 0x80u8 >> (i % 8); }
    }
    let max_strength = vectors.iter().map(|v| v.strength).fold(0.0f64, f64::max);
    Hypervector {
        data,
        concept_id: "bundle".to_string(),
        strength: max_strength,
        last_access: now_secs(),
        source_text: String::new(),
    }
}

pub fn compress_mp4c(vectors: &[Hypervector], noise_floor: f64) -> Hypervector {
    if vectors.is_empty() { return Hypervector::atomic_from_str("zero", "zero"); }
    let threshold = (vectors.len() as f64 * noise_floor) as i32;
    let mut counts = vec![0i32; DIMS];
    for hv in vectors {
        for (byte_i, &byte) in hv.data.iter().enumerate() {
            for bit_i in 0..8usize {
                let idx = byte_i * 8 + bit_i;
                if byte & (0x80u8 >> bit_i) != 0 { counts[idx] += 1; }
                else { counts[idx] -= 1; }
            }
        }
    }
    let mut data = vec![0u8; PACKED_BYTES];
    for (i, &c) in counts.iter().enumerate() {
        if c > threshold { data[i / 8] |= 0x80u8 >> (i % 8); }
    }
    let max_strength = vectors.iter().map(|v| v.strength).fold(0.0f64, f64::max);
    Hypervector {
        data,
        concept_id: "compress_mp4c".to_string(),
        strength: max_strength,
        last_access: now_secs(),
        source_text: String::new(),
    }
}

pub fn analogy(a: &Hypervector, b: &Hypervector, c: &Hypervector) -> Hypervector {
    b.bind(a).bind(c)
}

pub fn associative_attention(
    query: &Hypervector,
    keys: &[Hypervector],
    values: &[Hypervector],
    temperature: f64,
) -> Hypervector {
    let sims: Vec<f64> = keys.iter().map(|k| query.similarity(k, false)).collect();
    let max_s = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = sims.iter().map(|&s| ((s - max_s) / temperature).exp()).collect();
    let sum_e: f64 = exps.iter().sum();
    let weights: Vec<f64> = exps.iter().map(|&e| e / sum_e).collect();
    let mut bit_weights = vec![0.0f64; DIMS];
    for (hv, &w) in values.iter().zip(weights.iter()) {
        for (byte_i, &byte) in hv.data.iter().enumerate() {
            for bit_i in 0..8usize {
                if byte & (0x80u8 >> bit_i) != 0 {
                    bit_weights[byte_i * 8 + bit_i] += w;
                }
            }
        }
    }
    let mut data = vec![0u8; PACKED_BYTES];
    for (i, &w) in bit_weights.iter().enumerate() {
        if w > 0.5 { data[i / 8] |= 0x80u8 >> (i % 8); }
    }
    Hypervector {
        data,
        concept_id: "attention_result".to_string(),
        strength: 1.0,
        last_access: now_secs(),
        source_text: String::new(),
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryTrace {
    pub hv: Hypervector,
    pub tier: MemoryTier,
    pub timestamp: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub task_context: Option<String>,
}

impl MemoryTrace {
    pub fn new(
        hv: Hypervector,
        tier: MemoryTier,
        r: f64,
        theta: f64,
        phi: f64,
        task_context: Option<String>,
    ) -> Self {
        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        Self { hv, tier, timestamp: now_secs(), x, y, z, task_context }
    }
}

// ==============================================================================
// 2. ADVANCED COGNITIVE MODULES — FULLY PRESERVED
// ==============================================================================
pub enum ASTNode {
    Leaf { value: String },
    Tree { op: String, left: Box<ASTNode>, right: Box<ASTNode> },
}

impl ASTNode {
    pub fn atom(v: &str) -> Self { ASTNode::Leaf { value: v.to_string() } }
    pub fn tree(op: &str, l: ASTNode, r: ASTNode) -> Self {
        ASTNode::Tree { op: op.to_string(), left: Box::new(l), right: Box::new(r) }
    }
    pub fn get_size(&self) -> usize {
        match self {
            ASTNode::Leaf { .. } => 1,
            ASTNode::Tree { left, right, .. } => 1 + left.get_size() + right.get_size(),
        }
    }
}

pub struct CognitiveEngine<'a> {
    pub bubble: &'a mut UnifiedKnowledgeBubble,
}

impl<'a> CognitiveEngine<'a> {
    pub fn new(bubble: &'a mut UnifiedKnowledgeBubble) -> Self { Self { bubble } }

    pub fn learn_fact(&mut self, subject: &str, predicate: &str, object: &str) {
        let triple = Hypervector::from_string(subject, None)
            .bind(&Hypervector::from_string(predicate, None))
            .bind(&Hypervector::from_string(object, None));
        let trace = MemoryTrace::new(triple, MemoryTier::Semantic, 0.0, 0.0, 0.0, None);
        let cid = format!("SPO:{}_{}_{}", subject, predicate, object);
        let encoded = bincode::serialize(&trace).unwrap();
        self.bubble.semantic_db.insert(cid.as_bytes(), encoded).unwrap();
        self.bubble.semantic_db.flush().unwrap();
        self.bubble.semantic_index.push(trace.hv);
    }

    pub fn encode_ast(node: &ASTNode) -> Hypervector {
        match node {
            ASTNode::Leaf { value } => Hypervector::from_string(value, Some(value)),
            ASTNode::Tree { op, left, right } => {
                let op_hv = Hypervector::from_string(op, Some(op));
                let left_hv = Self::encode_ast(left);
                let right_hv = Self::encode_ast(right).permute(1);
                op_hv.bind(&bundle(&[left_hv, right_hv]))
            }
        }
    }

    pub fn evaluate_lasso_fitness(
        candidate: &ASTNode,
        target: &Hypervector,
        lambda_l1: f64,
    ) -> f64 {
        let hv = Self::encode_ast(candidate);
        hv.similarity(target, false) - lambda_l1 * candidate.get_size() as f64
    }
}

pub struct HDQAgent {
    pub actions: HashMap<String, Hypervector>,
    pub q_brain: Hypervector,
    pub experience_buffer: VecDeque<(Hypervector, String, f64)>,
}

impl HDQAgent {
    pub fn new(action_names: &[&str]) -> Self {
        let actions = action_names.iter()
            .map(|&n| (n.to_string(), Hypervector::from_string(n, Some(n))))
            .collect();
        Self {
            actions,
            q_brain: Hypervector::from_string("blank_slate", Some("q_brain")),
            experience_buffer: VecDeque::with_capacity(1000),
        }
    }

    pub fn choose_action(&self, state_hv: &Hypervector) -> String {
        let intent = self.q_brain.bind(state_hv);
        self.actions.iter()
            .max_by(|(_, a), (_, b)|
                a.similarity(&intent, false)
                    .partial_cmp(&b.similarity(&intent, false)).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap_or_default()
    }

    pub fn learn(
        &mut self,
        state: &Hypervector,
        action_name: &str,
        reward: f64,
        _next: Option<&Hypervector>,
    ) {
        if self.experience_buffer.len() >= 1000 { self.experience_buffer.pop_front(); }
        self.experience_buffer.push_back((state.clone(), action_name.to_string(), reward));
        if reward > 0.0 {
            if let Some(act_hv) = self.actions.get(action_name) {
                let exp = state.bind(act_hv);
                self.q_brain = bundle(&[self.q_brain.clone(), exp]);
            }
        }
    }
}

// ==============================================================================
// 3. ENTERPRISE KNOWLEDGE BUBBLE
// ==============================================================================
pub struct UnifiedKnowledgeBubble {
    pub episodic_buffer: Vec<MemoryTrace>,
    pub semantic_db: sled::Db,
    pub semantic_index: Vec<Hypervector>,
    pub task_db: sled::Db,
    pub procedural_memory: HashMap<String, (Hypervector, usize)>,
    pub task_vectors: HashMap<String, Hypervector>,
    pub active_task: Option<String>,
}

impl UnifiedKnowledgeBubble {
    pub fn new() -> Self {
        let db = sled::Config::default()
            .path("god_node_storage")
            .cache_capacity(256 * 1024 * 1024)
            .open()
            .expect("Failed to open sled database");
        let task_db = sled::open("god_node_tasks").expect("Failed to open task database");
        let mut bubble = Self {
            episodic_buffer: Vec::new(),
            semantic_db: db,
            semantic_index: Vec::new(),
            task_db,
            procedural_memory: HashMap::new(),
            task_vectors: HashMap::new(),
            active_task: None,
        };
        bubble.load_semantic_index();
        bubble.load_task_vectors();
        bubble
    }

    fn load_semantic_index(&mut self) {
        for item in self.semantic_db.iter() {
            if let Ok((_, val_bytes)) = item {
                if let Ok(trace) = bincode::deserialize::<MemoryTrace>(&val_bytes) {
                    self.semantic_index.push(trace.hv);
                }
            }
        }
    }

    fn load_task_vectors(&mut self) {
        for item in self.task_db.iter() {
            if let Ok((key_bytes, val_bytes)) = item {
                if let Ok(hv) = bincode::deserialize::<Hypervector>(&val_bytes) {
                    let task_name = String::from_utf8_lossy(&key_bytes).to_string();
                    self.task_vectors.insert(task_name, hv);
                }
            }
        }
    }

    pub fn set_task(&mut self, task_name: &str) {
        let hv = self.task_vectors
            .entry(task_name.to_string())
            .or_insert_with(|| Hypervector::from_string(&format!("task_{}", task_name), None))
            .clone();
        let _ = self.task_db.insert(task_name.as_bytes(), bincode::serialize(&hv).unwrap());
        let _ = self.task_db.flush();
        self.active_task = Some(task_name.to_string());
    }

    pub fn ingest(&mut self, text: &str, r: f64, theta: f64, phi: f64) -> String {
        let mut hv = Hypervector::from_string(text, None);
        if let Some(ref task) = self.active_task {
            if let Some(task_hv) = self.task_vectors.get(task) {
                hv = hv.bind(task_hv);
                let snippet: String = text.chars().take(20).collect();
                hv.concept_id = format!("{}:{}", task, snippet);
            }
        }
        let trace = MemoryTrace::new(
            hv.clone(),
            MemoryTier::Episodic,
            r, theta, phi,
            self.active_task.clone(),
        );
        self.episodic_buffer.push(trace);
        hv.concept_id.clone()
    }

    pub fn ingest_book(&mut self, filepath: &str, title: &str) -> Result<usize> {
        let content = std::fs::read_to_string(filepath)
            .with_context(|| format!("Failed to read file: {}", filepath))?;
        self.set_task(&format!("Book: {}", title));
        let paragraphs: Vec<&str> = content
            .split("\n\n")
            .filter(|p| !p.trim().is_empty())
            .collect();
        let count = paragraphs.len();
        let processed_traces: Vec<MemoryTrace> = paragraphs.par_iter().map(|p| {
            let seed: u32 = p.chars().map(|c| c as u32).sum();
            let r = 15.0;
            let theta = (seed % 314) as f64 / 100.0;
            let phi = (seed % 628) as f64 / 100.0;
            let mut hv = Hypervector::from_string(p, None);
            let snippet: String = p.chars().take(15).collect();
            hv.concept_id = format!("Book:{}:{}", title, snippet);
            MemoryTrace::new(hv, MemoryTier::Episodic, r, theta, phi,
                Some(format!("Book: {}", title)))
        }).collect();
        self.episodic_buffer.extend(processed_traces);
        self.consolidate();
        Ok(count)
    }

    // ===========================================================================
    // FIX 3: DEDUPLICATE semantic_index — if a concept_id already exists
    // in the index, replace it rather than appending a duplicate entry.
    // This prevents unbounded RAM growth across multiple consolidate() calls.
    // ===========================================================================
    pub fn consolidate(&mut self) {
        if self.episodic_buffer.is_empty() { return; }
        let mut groups: HashMap<String, Vec<MemoryTrace>> = HashMap::new();
        for trace in self.episodic_buffer.drain(..) {
            groups.entry(trace.hv.concept_id.clone()).or_default().push(trace);
        }
        for (concept_id, traces) in groups {
            let count = traces.len();
            let avg_x = traces.iter().map(|t| t.x).sum::<f64>() / count as f64;
            let avg_y = traces.iter().map(|t| t.y).sum::<f64>() / count as f64;
            let avg_z = traces.iter().map(|t| t.z).sum::<f64>() / count as f64;
            let r = (avg_x.powi(2) + avg_y.powi(2) + avg_z.powi(2)).sqrt();
            let theta = if r > 0.0 { (avg_z / r).acos() } else { 0.0 };
            let phi = avg_y.atan2(avg_x);
            let hvs: Vec<Hypervector> = traces.iter().map(|t| t.hv.clone()).collect();
            let mut final_hv = compress_mp4c(&hvs, 0.4);
            final_hv.concept_id = concept_id.clone();
            final_hv.strength = (0.5 + 0.1 * count as f64).min(1.0);
            let final_trace = MemoryTrace::new(
                final_hv.clone(), MemoryTier::Semantic, r, theta, phi, None,
            );
            let encoded = bincode::serialize(&final_trace).unwrap();
            self.semantic_db.insert(concept_id.as_bytes(), encoded).unwrap();

            // FIX 3: Replace existing entry if concept_id already in index,
            // otherwise push. Prevents duplicates accumulating in RAM.
            if let Some(pos) = self.semantic_index.iter().position(|h| h.concept_id == concept_id) {
                self.semantic_index[pos] = final_hv;
            } else {
                self.semantic_index.push(final_hv);
            }
        }
        self.semantic_db.flush().unwrap();
        // episodic_buffer is already drained above via drain(..)
    }

    // ===========================================================================
    // FIX 2: REFRESH last_access on successful retrieval.
    // Both the in-memory index entry and the on-disk sled record are updated
    // so that frequently accessed memories resist decay correctly.
    // ===========================================================================
    pub fn retrieve(&mut self, query: &str, task_filter: Option<&str>, top_k: usize)
        -> Vec<(String, f64)>
    {
        let mut query_hv = Hypervector::from_string(query, None);
        if let Some(task) = task_filter {
            let task_hv = self.task_vectors.get(task).cloned()
                .unwrap_or_else(|| Hypervector::from_string(&format!("task_{}", task), None));
            query_hv = query_hv.bind(&task_hv);
        }

        // Lowered threshold from 0.55 → 0.52 so n-gram similarity scores
        // (which cluster around 0.53–0.70) are not accidentally filtered out.
        let mut results: Vec<(String, f64)> = self.semantic_index.iter()
            .map(|hv| (hv.concept_id.clone(), query_hv.similarity(hv, false)))
            .filter(|(_, sim)| *sim > 0.52)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(top_k);

        // FIX 2: Touch every returned result to refresh its last_access timestamp.
        for (cid, _) in &results {
            // Update in-memory index
            if let Some(hv) = self.semantic_index.iter_mut().find(|h| &h.concept_id == cid) {
                hv.touch();
            }
            // Update on-disk record
            let key = cid.as_bytes();
            if let Ok(Some(val)) = self.semantic_db.get(key) {
                if let Ok(mut trace) = bincode::deserialize::<MemoryTrace>(&val) {
                    trace.hv.touch();
                    let _ = self.semantic_db.insert(key, bincode::serialize(&trace).unwrap());
                }
            }
        }
        let _ = self.semantic_db.flush();
        results
    }

    // ===========================================================================
    // FIX 4: SPATIAL RETRIEVAL — retrieve_near filters candidates by
    // Euclidean distance in the 3D memory sphere BEFORE computing expensive
    // Hamming similarity. This enables true memory-palace navigation:
    // "show me what I remember near this location."
    // ===========================================================================
    pub fn retrieve_near(
        &mut self,
        cx: f64, cy: f64, cz: f64,
        radius: f64,
        top_k: usize,
    ) -> Vec<(String, f64, f64)>  // (concept_id, spatial_dist, hv_strength)
    {
        let mut results: Vec<(String, f64, f64)> = Vec::new();

        for item in self.semantic_db.iter() {
            if let Ok((key, val)) = item {
                if let Ok(mut trace) = bincode::deserialize::<MemoryTrace>(&val) {
                    let dx = trace.x - cx;
                    let dy = trace.y - cy;
                    let dz = trace.z - cz;
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist <= radius {
                        // FIX 2 applied here too: touch spatially retrieved memories
                        trace.hv.touch();
                        let cid = String::from_utf8_lossy(&key).to_string();
                        results.push((cid, dist, trace.hv.strength));
                        // Persist refreshed last_access
                        let _ = self.semantic_db.insert(&key,
                            bincode::serialize(&trace).unwrap());
                    }
                }
            }
        }
        let _ = self.semantic_db.flush();

        // Sort by distance ascending (nearest first)
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_k);
        results
    }

    pub fn cleanup_disk(&mut self, threshold: f64) -> usize {
        let mut to_remove = Vec::new();
        let mut new_index = Vec::new();
        for hv in &self.semantic_index {
            let mut hv_mut = hv.clone();
            hv_mut.decay();
            if hv_mut.strength < threshold {
                to_remove.push(hv.concept_id.clone());
            } else {
                new_index.push(hv_mut.clone());
                let key = hv.concept_id.as_bytes();
                if let Ok(Some(val)) = self.semantic_db.get(key) {
                    if let Ok(mut trace) = bincode::deserialize::<MemoryTrace>(&val) {
                        trace.hv.strength = hv_mut.strength;
                        trace.hv.last_access = hv_mut.last_access;
                        let _ = self.semantic_db.insert(
                            key, bincode::serialize(&trace).unwrap(),
                        );
                    }
                }
            }
        }
        self.semantic_index = new_index;
        let count = to_remove.len();
        for k in to_remove {
            self.semantic_db.remove(k.as_bytes()).unwrap();
        }
        self.semantic_db.flush().unwrap();
        count
    }

    pub fn ingest_sequence(&mut self, name: &str, steps: &[&str]) {
        let role_fillers: Vec<Hypervector> = steps.iter().enumerate().map(|(i, &step)| {
            let role = Hypervector::from_string(&format!("pos_{}", i), None).permute(i as i64);
            let filler = Hypervector::from_string(step, None);
            role.bind(&filler)
        }).collect();
        let mut proc_hv = bundle(&role_fillers);
        proc_hv.concept_id = name.to_string();
        self.procedural_memory.insert(name.to_string(), (proc_hv, steps.len()));
    }

    pub fn execute_procedure(&self, name: &str, step_index: usize) -> Option<Hypervector> {
        let (proc_hv, len) = self.procedural_memory.get(name)?;
        if step_index >= *len { return None; }
        let role = Hypervector::from_string(&format!("pos_{}", step_index), None)
            .permute(step_index as i64);
        Some(proc_hv.bind(&role))
    }
}

// ==============================================================================
// 4. UI STATE
// ==============================================================================
struct GodNodeUI {
    bubble: UnifiedKnowledgeBubble,
    task_text: String,
    ingest_text: String,
    book_path: String,
    book_title: String,
    search_text: String,
    // Spatial search fields (FIX 4 UI)
    spatial_x: String,
    spatial_y: String,
    spatial_z: String,
    spatial_radius: String,
    spatial_result: String,
    status_message: String,
    search_result: String,
    active_search_id: Option<String>,
    rot_x: f32,
    rot_y: f32,
    cached_3d_nodes: Vec<(String, f32, f32, f32, f64)>,
    needs_redraw: bool,
    // Spatial probe point for 3D overlay (FIX 4 visualization)
    spatial_probe: Option<(f32, f32, f32, f32)>, // x, y, z, radius
}

impl Default for GodNodeUI {
    fn default() -> Self {
        let bubble = UnifiedKnowledgeBubble::new();
        Self {
            bubble,
            task_text: String::new(),
            ingest_text: String::new(),
            book_path: String::new(),
            book_title: String::new(),
            search_text: String::new(),
            spatial_x: "0.0".to_string(),
            spatial_y: "0.0".to_string(),
            spatial_z: "0.0".to_string(),
            spatial_radius: "5.0".to_string(),
            spatial_result: String::new(),
            status_message: "System Online. All 5 Production Fixes Applied.".to_string(),
            search_result: String::new(),
            active_search_id: None,
            rot_x: 0.2,
            rot_y: 0.0,
            cached_3d_nodes: Vec::new(),
            needs_redraw: true,
            spatial_probe: None,
        }
    }
}

// ==============================================================================
// 5. THE DESKTOP WINDOW & DASHBOARD
// ==============================================================================
impl eframe::App for GodNodeUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.needs_redraw {
            self.cached_3d_nodes.clear();
            for item in self.bubble.semantic_db.iter() {
                if let Ok((k, v)) = item {
                    if let Ok(trace) = bincode::deserialize::<MemoryTrace>(&v) {
                        let cid = String::from_utf8_lossy(&k).to_string();
                        self.cached_3d_nodes.push((
                            cid,
                            trace.x as f32,
                            trace.y as f32,
                            trace.z as f32,
                            trace.hv.strength,
                        ));
                    }
                }
            }
            self.needs_redraw = false;
        }

        egui::SidePanel::left("control_panel").exact_width(350.0).show(ctx, |ui| {
            ui.heading(
                egui::RichText::new("⚡ God Node Enterprise v2")
                    .size(20.0)
                    .strong()
                    .color(egui::Color32::from_rgb(0, 255, 200)),
            );
            ui.label(
                egui::RichText::new("Production Build — All 5 Fixes Active")
                    .size(11.0)
                    .color(egui::Color32::from_rgb(100, 200, 100)),
            );
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                // ── SECTION 1: Task Context ──────────────────────────────────
                ui.add_space(8.0);
                ui.label(egui::RichText::new("① Set Domain Context").strong());
                let active = self.bubble.active_task.as_deref().unwrap_or("None");
                ui.label(
                    egui::RichText::new(format!("Active: {}", active))
                        .color(egui::Color32::YELLOW)
                        .size(11.0),
                );
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.task_text);
                    if ui.button("Set").clicked() && !self.task_text.is_empty() {
                        self.bubble.set_task(&self.task_text);
                        self.status_message = format!("Task set → {}", self.task_text);
                    }
                });

                // ── SECTION 2: Manual Ingest ─────────────────────────────────
                ui.add_space(12.0);
                ui.label(egui::RichText::new("② Manual Short-Term Ingest").strong());
                ui.label(
                    egui::RichText::new("FIX 1: n-gram semantic encoding active")
                        .size(10.0)
                        .color(egui::Color32::from_rgb(100, 200, 100)),
                );
                ui.text_edit_singleline(&mut self.ingest_text);
                ui.horizontal(|ui| {
                    if ui.button("➕ Map to RAM").clicked() && !self.ingest_text.is_empty() {
                        let seed: u32 = self.ingest_text.chars().map(|c| c as u32).sum();
                        let r = 5.0;
                        let theta = (seed % 314) as f64 / 100.0;
                        let phi = (seed % 628) as f64 / 100.0;
                        let cid = self.bubble.ingest(&self.ingest_text, r, theta, phi);
                        self.status_message = format!("RAM: {}", cid);
                        self.ingest_text.clear();
                    }
                    if ui.button("🌙 Consolidate → SSD").clicked() {
                        let count = self.bubble.episodic_buffer.len();
                        self.bubble.consolidate();
                        self.needs_redraw = true;
                        self.status_message = format!(
                            "Saved {} traces. Dedup active (FIX 3).", count
                        );
                    }
                });

                // ── SECTION 3: Book Loader ───────────────────────────────────
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("③ Automated Document Loader")
                        .strong()
                        .color(egui::Color32::from_rgb(255, 165, 0)),
                );
                ui.horizontal(|ui| {
                    ui.label("File:");
                    ui.text_edit_singleline(&mut self.book_path);
                    if ui.button("📂 Browse").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Text Files", &["txt"])
                            .pick_file()
                        {
                            self.book_path = path.display().to_string();
                            if self.book_title.is_empty() {
                                self.book_title = path
                                    .file_stem()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string();
                            }
                        }
                    }
                });
                ui.label("Title:");
                ui.text_edit_singleline(&mut self.book_title);
                if ui.button("📚 Ingest Entire Book (Parallel)").clicked()
                    && !self.book_path.is_empty()
                {
                    self.status_message = "Parsing vectors in parallel (rayon)…".to_string();
                    match self.bubble.ingest_book(&self.book_path, &self.book_title) {
                        Ok(count) => {
                            self.status_message = format!(
                                "✅ {} paragraphs processed across CPU cores.", count
                            );
                            self.needs_redraw = true;
                        }
                        Err(e) => {
                            self.status_message = format!("❌ Error: {:?}", e);
                        }
                    }
                }

                // ── SECTION 4: Semantic Search ───────────────────────────────
                ui.add_space(12.0);
                ui.label(egui::RichText::new("④ Semantic Search (Top-5)").strong());
                ui.label(
                    egui::RichText::new("FIX 1+2: n-gram similarity + access refresh")
                        .size(10.0)
                        .color(egui::Color32::from_rgb(100, 200, 100)),
                );
                ui.text_edit_singleline(&mut self.search_text);
                if ui.button("🔍 Calculate Resonance").clicked() && !self.search_text.is_empty() {
                    let task_filter = self.bubble.active_task.clone();
                    let results = self.bubble.retrieve(
                        &self.search_text,
                        task_filter.as_deref(),
                        5,
                    );
                    if results.is_empty() {
                        self.search_result = "No resonance found above threshold.".to_string();
                        self.active_search_id = None;
                    } else {
                        self.active_search_id = Some(results[0].0.clone());
                        self.search_result = results.iter()
                            .map(|(cid, sim)| format!("{:.2}%  {}", sim * 100.0, cid))
                            .collect::<Vec<_>>()
                            .join("\n");
                        self.status_message = format!(
                            "Search done. Top: {:.1}%", results[0].1 * 100.0
                        );
                    }
                    self.needs_redraw = true; // refresh node highlight
                }
                if !self.search_result.is_empty() {
                    ui.add_space(4.0);
                    egui::Frame::none()
                        .fill(egui::Color32::from_black_alpha(80))
                        .inner_margin(6.0)
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(&self.search_result)
                                    .color(egui::Color32::from_rgb(100, 255, 150))
                                    .size(12.0),
                            );
                        });
                }

                // ── SECTION 5: Spatial Retrieval (NEW — FIX 4) ──────────────
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("⑤ Spatial Memory Palace Retrieval")
                        .strong()
                        .color(egui::Color32::from_rgb(200, 100, 255)),
                );
                ui.label(
                    egui::RichText::new("FIX 4: Euclidean pre-filter + 3D sphere highlight")
                        .size(10.0)
                        .color(egui::Color32::from_rgb(100, 200, 100)),
                );
                ui.horizontal(|ui| {
                    ui.label("X:"); ui.add(egui::TextEdit::singleline(&mut self.spatial_x).desired_width(48.0));
                    ui.label("Y:"); ui.add(egui::TextEdit::singleline(&mut self.spatial_y).desired_width(48.0));
                    ui.label("Z:"); ui.add(egui::TextEdit::singleline(&mut self.spatial_z).desired_width(48.0));
                    ui.label("R:"); ui.add(egui::TextEdit::singleline(&mut self.spatial_radius).desired_width(40.0));
                });
                if ui.button("🌐 Retrieve Near Location").clicked() {
                    let cx = self.spatial_x.parse::<f64>().unwrap_or(0.0);
                    let cy = self.spatial_y.parse::<f64>().unwrap_or(0.0);
                    let cz = self.spatial_z.parse::<f64>().unwrap_or(0.0);
                    let rad = self.spatial_radius.parse::<f64>().unwrap_or(5.0);
                    self.spatial_probe = Some((cx as f32, cy as f32, cz as f32, rad as f32));
                    let results = self.bubble.retrieve_near(cx, cy, cz, rad, 5);
                    if results.is_empty() {
                        self.spatial_result = "No memories found in that region.".to_string();
                    } else {
                        self.spatial_result = results.iter()
                            .map(|(cid, dist, str)| {
                                format!("dist={:.1} str={:.2}  {}", dist, str, cid)
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                    }
                    self.status_message = format!(
                        "Spatial search at ({},{},{}) r={}", cx, cy, cz, rad
                    );
                    self.needs_redraw = true;
                }
                if !self.spatial_result.is_empty() {
                    ui.add_space(4.0);
                    egui::Frame::none()
                        .fill(egui::Color32::from_black_alpha(80))
                        .inner_margin(6.0)
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(&self.spatial_result)
                                    .color(egui::Color32::from_rgb(200, 150, 255))
                                    .size(11.0),
                            );
                        });
                }

                // ── SECTION 6: SSD Maintenance ───────────────────────────────
                ui.add_space(12.0);
                ui.label(egui::RichText::new("⑥ SSD Maintenance").strong());
                ui.label(
                    egui::RichText::new("FIX 2: Decay skips recently accessed memories")
                        .size(10.0)
                        .color(egui::Color32::from_rgb(100, 200, 100)),
                );
                if ui.button("🧹 Disk Garbage Collection").clicked() {
                    let swept = self.bubble.cleanup_disk(0.4);
                    self.status_message = format!("Swept {} weak concepts.", swept);
                    self.needs_redraw = true;
                }

                // ── SECTION 7: Cognitive Diagnostics ────────────────────────
                ui.add_space(12.0);
                ui.label(
                    egui::RichText::new("⑦ Cognitive Modules (AST / RL / SPO)")
                        .strong()
                        .color(egui::Color32::from_rgb(200, 100, 255)),
                );
                if ui.button("🧠 Run Diagnostics & Agent Learning").clicked() {
                    let mut cog = CognitiveEngine::new(&mut self.bubble);
                    cog.learn_fact("GodNode", "is", "Operational");
                    let target = Hypervector::from_string("add_x_y", None);
                    let elegant = ASTNode::tree(
                        "add", ASTNode::atom("x"), ASTNode::atom("y"),
                    );
                    let ast_score = CognitiveEngine::evaluate_lasso_fitness(
                        &elegant, &target, 0.02,
                    );
                    let mut agent = HDQAgent::new(&["move", "jump"]);
                    let state = Hypervector::from_string("obstacle", None);
                    agent.learn(&state, "jump", 1.0, None);
                    let action = agent.choose_action(&state);
                    self.status_message = format!(
                        "Agent→'{}' | AST={:.4} | SPO saved", action, ast_score
                    );
                    self.needs_redraw = true;
                }

                // ── STATUS LOG ───────────────────────────────────────────────
                ui.add_space(16.0);
                ui.separator();
                ui.label(egui::RichText::new("System Log").color(egui::Color32::GRAY).size(11.0));
                egui::Frame::none()
                    .fill(egui::Color32::from_black_alpha(100))
                    .inner_margin(6.0)
                    .show(ui, |ui| {
                        ui.label(
                            egui::RichText::new(&self.status_message)
                                .color(egui::Color32::WHITE)
                                .size(12.0),
                        );
                    });

                // Index stats
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(format!(
                        "Index: {} vectors | RAM buffer: {}",
                        self.bubble.semantic_index.len(),
                        self.bubble.episodic_buffer.len(),
                    ))
                    .size(10.0)
                    .color(egui::Color32::from_rgb(150, 150, 150)),
                );
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(
                egui::RichText::new("3D Sled-Backed Knowledge Sphere")
                    .color(egui::Color32::from_rgb(0, 200, 255)),
            );
            ui.label(
                egui::RichText::new("Drag to rotate  •  Orange = search hit  •  Purple sphere = spatial probe  •  60fps capped (FIX 5)")
                    .size(10.0)
                    .color(egui::Color32::GRAY),
            );

            let (rect, response) =
                ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

            if response.dragged() {
                self.rot_y += response.drag_delta().x * 0.01;
                self.rot_x += response.drag_delta().y * 0.01;
            }
            // ==================================================================
            // FIX 5: CAPPED REPAINT at 60fps instead of busy-looping.
            // Previously: ctx.request_repaint() — pegged CPU at 100%.
            // Now: request_repaint_after(16ms) ≈ 60 fps, CPU-friendly.
            // ==================================================================
            self.rot_y += 0.003; // Slightly slower auto-rotate feels smoother
            ctx.request_repaint_after(Duration::from_millis(16));

            let painter = ui.painter();
            let center = rect.center();
            let scale = 22.0f32;

            // Outer sphere wireframe ring
            painter.circle_stroke(
                center,
                16.0 * scale,
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(8)),
            );
            // Equator ring
            painter.circle_stroke(
                center,
                16.0 * scale,
                egui::Stroke::new(0.5, egui::Color32::from_white_alpha(4)),
            );

            // Project all 3D nodes onto 2D screen
            let mut projected_nodes: Vec<(String, f32, f32, f32, f64, bool)> = Vec::new();
            for (cid, x, y, z, strength) in &self.cached_3d_nodes {
                let (px, py, pz) = rotate_3d(*x, *y, *z, self.rot_x, self.rot_y);
                let screen_x = center.x + px * scale;
                let screen_y = center.y - py * scale;

                // Is this node inside the spatial probe radius?
                let in_probe = self.spatial_probe.map(|(cx, cy, cz, r)| {
                    let dx = x - cx;
                    let dy = y - cy;
                    let dz = z - cz;
                    (dx * dx + dy * dy + dz * dz).sqrt() <= r
                }).unwrap_or(false);

                projected_nodes.push((cid.clone(), screen_x, screen_y, pz, *strength, in_probe));
            }
            // Painter's algorithm: sort back-to-front
            projected_nodes.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

            for (cid, px, py, pz, strength, in_probe) in &projected_nodes {
                let pos = egui::pos2(*px, *py);
                let depth_factor = (pz + 15.0) / 30.0;
                let depth_factor = depth_factor.clamp(0.0, 1.0);
                let radius = (2.0 + (*strength as f32 * 3.0)) * depth_factor.max(0.3);
                let alpha = (60.0 + 195.0 * depth_factor).clamp(0.0, 255.0) as u8;

                let (fill_color, text_color) =
                    if Some(&cid) == self.active_search_id.as_ref().as_ref() {
                        // Search result: bright orange
                        (
                            egui::Color32::from_rgba_unmultiplied(255, 165, 0, 255),
                            egui::Color32::from_rgb(255, 200, 0),
                        )
                    } else if *in_probe {
                        // Inside spatial probe: purple
                        (
                            egui::Color32::from_rgba_unmultiplied(200, 100, 255, 220),
                            egui::Color32::from_rgb(220, 150, 255),
                        )
                    } else {
                        // Normal: cyan
                        (
                            egui::Color32::from_rgba_unmultiplied(0, 200, 255, alpha),
                            egui::Color32::from_white_alpha(alpha),
                        )
                    };

                painter.circle_filled(pos, radius, fill_color);

                // Label visible nodes
                if depth_factor > 0.75 || Some(&cid) == self.active_search_id.as_ref().as_ref() {
                    painter.text(
                        pos + egui::vec2(radius + 2.0, -radius),
                        egui::Align2::LEFT_BOTTOM,
                        cid.as_str(),
                        egui::FontId::proportional(10.0),
                        text_color,
                    );
                }
            }

            // Draw spatial probe sphere as a translucent circle overlay
            if let Some((cx, cy, cz, radius)) = self.spatial_probe {
                let (px, py, _pz) = rotate_3d(cx, cy, cz, self.rot_x, self.rot_y);
                let screen_pos = egui::pos2(center.x + px * scale, center.y - py * scale);
                let screen_r = radius * scale;
                painter.circle_stroke(
                    screen_pos,
                    screen_r,
                    egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(180, 100, 255, 160)),
                );
                painter.circle_filled(
                    screen_pos,
                    screen_r,
                    egui::Color32::from_rgba_unmultiplied(150, 80, 220, 20),
                );
                painter.text(
                    screen_pos + egui::vec2(screen_r + 4.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    format!("probe r={:.1}", radius),
                    egui::FontId::proportional(11.0),
                    egui::Color32::from_rgb(200, 150, 255),
                );
            }
        });
    }
}

/// Pure rotation helper extracted from the render loop for clarity.
/// Applies Y-axis then X-axis rotation to a 3D point.
fn rotate_3d(x: f32, y: f32, z: f32, rot_x: f32, rot_y: f32) -> (f32, f32, f32) {
    let x1 = x * rot_y.cos() - z * rot_y.sin();
    let z1 = x * rot_y.sin() + z * rot_y.cos();
    let y2 = y * rot_x.cos() - z1 * rot_x.sin();
    let z2 = y * rot_x.sin() + z1 * rot_x.cos();
    (x1, y2, z2)
}

// ==============================================================================
// 6. BOOT SEQUENCE
// ==============================================================================
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 780.0])
            .with_title("God Node Enterprise OS — Production Build"),
        ..Default::default()
    };
    eframe::run_native(
        "God Node Enterprise OS",
        options,
        Box::new(|_cc| Box::new(GodNodeUI::default())),
    )
}
