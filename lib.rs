#![allow(dead_code, unused_imports)]
use std::collections::{HashMap, VecDeque};
use sha2::{Sha256, Digest};
pub const DIMS: usize         = 8192;
pub const PACKED_BYTES: usize = DIMS / 8;
pub const DECAY_RATE: f64     = 0.01;
pub fn now_secs() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryTier { Working, Episodic, Semantic, Procedural }
#[derive(Clone)]
pub struct Hypervector {
    pub data: Vec<u8>,
    pub concept_id: String,
    pub strength: f64,
    pub last_access: f64,
    pub source_text: String,
}
impl Hypervector {
    pub fn from_string(text: &str, concept_id: Option<&str>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let digest = hasher.finalize();
        let seed = u64::from_le_bytes([
            digest[0], digest[1], digest[2], digest[3],
            digest[4], digest[5], digest[6], digest[7],
        ]);
        let mut state = seed;
        let data: Vec<u8> = (0..PACKED_BYTES).map(|_| xorshift64(&mut state)).collect();
        let concept_id = concept_id.map(|s| s.to_string())
            .unwrap_or_else(|| text.chars().take(50).collect());
        Hypervector {
            data,
            concept_id,
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
            concept_id: format!("bound_{}_{}", self.concept_id, other.concept_id),
            strength: self.strength.min(other.strength) * 0.95,
            last_access: now_secs(),
            source_text: self.source_text.clone(),
        }
    }
    pub fn permute(&self, shift: i64) -> Self {
        let shift = ((shift % 8192 + 8192) as usize) % 8192;
        let byte_shift = shift / 8;
        let bit_shift  = shift % 8;
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
        let bit_shift  = shift % 8;
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
    pub fn access(&mut self, boost: f64) {
        self.strength = (self.strength + boost).min(2.0);
        self.last_access = now_secs();
    }
    pub fn decay(&mut self) {
        let e = now_secs() - self.last_access;
        self.strength *= (-DECAY_RATE * e).exp();
    }
    pub fn copy(&self) -> Self { self.clone() }
    pub fn to_json(&self) -> String {
        use base64::{Engine, engine::general_purpose::STANDARD};
        let encoded = STANDARD.encode(&self.data);
        serde_json::to_string(&(
            encoded, &self.concept_id, self.strength,
            self.last_access, &self.source_text
        )).unwrap()
    }
    pub fn from_json(s: &str) -> Self {
        use base64::{Engine, engine::general_purpose::STANDARD};
        let (encoded, concept_id, strength, last_access, source_text):
            (String, String, f64, f64, String) = serde_json::from_str(s).unwrap();
        let data = STANDARD.decode(&encoded).unwrap();
        Hypervector { data, concept_id, strength, last_access, source_text }
    }
}
fn xorshift64(state: &mut u64) -> u8 {
    let mut x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x; x as u8
}
pub fn bundle(vectors: &[Hypervector]) -> Hypervector {
    if vectors.is_empty() { return Hypervector::from_string("zero", None); }
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
    Hypervector { data, concept_id: "bundle".to_string(),
        strength: max_strength, last_access: now_secs(), source_text: String::new() }
}
pub fn compress_mp4c(vectors: &[Hypervector], noise_floor: f64) -> Hypervector {
    if vectors.is_empty() { return Hypervector::from_string("zero", None); }
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
    Hypervector { data, concept_id: "compress_mp4c".to_string(),
        strength: max_strength, last_access: now_secs(), source_text: String::new() }
}
pub fn analogy(a: &Hypervector, b: &Hypervector, c: &Hypervector) -> Hypervector {
    b.bind(a).bind(c)
}
pub fn associative_attention(
    query: &Hypervector, keys: &[Hypervector],
    values: &[Hypervector], temperature: f64,
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
    Hypervector { data, concept_id: "attention_result".to_string(),
        strength: 1.0, last_access: now_secs(), source_text: String::new() }
}
#[derive(Clone)]
pub struct MemoryTrace {
    pub hv: Hypervector,
    pub tier: MemoryTier,
    pub timestamp: f64,
    pub x: f64, pub y: f64, pub z: f64,
    pub task_context: Option<String>,
}
impl MemoryTrace {
    pub fn new(hv: Hypervector, tier: MemoryTier, r: f64, theta: f64, phi: f64,
               task_context: Option<String>) -> Self {
        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        Self { hv, tier, timestamp: now_secs(), x, y, z, task_context }
    }
    pub fn distance_to(&self, x: f64, y: f64, z: f64) -> f64 {
        ((self.x-x).powi(2)+(self.y-y).powi(2)+(self.z-z).powi(2)).sqrt()
    }
}
pub struct UnifiedKnowledgeBubble {
    pub episodic_buffer: Vec<MemoryTrace>,
    pub semantic_memory: HashMap<String, MemoryTrace>,
    pub procedural_memory: HashMap<String, (Hypervector, usize)>,
    pub task_vectors: HashMap<String, Hypervector>,
    pub active_task: Option<String>,
}
impl UnifiedKnowledgeBubble {
    pub fn new() -> Self {
        Self {
            episodic_buffer: Vec::new(),
            semantic_memory: HashMap::new(),
            procedural_memory: HashMap::new(),
            task_vectors: HashMap::new(),
            active_task: None,
        }
    }
    pub fn set_task(&mut self, task_name: &str) {
        self.task_vectors.entry(task_name.to_string())
            .or_insert_with(|| Hypervector::from_string(
                &format!("task_{}", task_name), None));
        self.active_task = Some(task_name.to_string());
    }
    pub fn ingest(&mut self, text: &str, r: f64, theta: f64, phi: f64) -> String {
        let mut hv = Hypervector::from_string(text, None);
        if let Some(ref task) = self.active_task {
            if let Some(task_hv) = self.task_vectors.get(task) {
                hv = hv.bind(task_hv);
                hv.concept_id = format!("{}:{}", task, &text[..text.len().min(20)]);
            }
        }
        let trace = MemoryTrace::new(hv.clone(), MemoryTier::Episodic,
            r, theta, phi, self.active_task.clone());
        self.episodic_buffer.push(trace);
        hv.concept_id.clone()
    }
    pub fn consolidate(&mut self) {
        let mut groups: HashMap<String, Vec<&MemoryTrace>> = HashMap::new();
        for trace in &self.episodic_buffer {
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
            let mut final_hv = bundle(&hvs);
            final_hv.concept_id = concept_id.clone();
            final_hv.strength = (0.5 + 0.1 * count as f64).min(1.0);
            self.semantic_memory.insert(concept_id,
                MemoryTrace::new(final_hv, MemoryTier::Semantic, r, theta, phi, None));
        }
        self.episodic_buffer.clear();
    }
    pub fn cleanup(&mut self, threshold: f64) -> usize {
        let mut to_remove = Vec::new();
        for (key, trace) in &mut self.semantic_memory {
            trace.hv.decay();
            if trace.hv.strength < threshold { to_remove.push(key.clone()); }
        }
        for key in &to_remove { self.semantic_memory.remove(key); }
        to_remove.len()
    }
    pub fn spatial_query(&self, r: f64, theta: f64, phi: f64,
                         radius: f64) -> Vec<(String, f64)> {
        let x = r * theta.sin() * phi.cos();
        let y = r * theta.sin() * phi.sin();
        let z = r * theta.cos();
        let mut results: Vec<(String, f64)> = self.semantic_memory.iter()
            .filter_map(|(cid, t)| {
                let d = t.distance_to(x, y, z);
                if d <= radius { Some((cid.clone(), d)) } else { None }
            }).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }
    pub fn retrieve(&self, query: &str, task_filter: Option<&str>) -> Vec<(String, f64)> {
        let mut query_hv = Hypervector::from_string(query, None);
        if let Some(task) = task_filter {
            if let Some(task_hv) = self.task_vectors.get(task) {
                query_hv = query_hv.bind(task_hv);
            }
        }
        let mut results: Vec<(String, f64)> = self.semantic_memory.iter()
            .filter_map(|(cid, t)| {
                let sim = query_hv.similarity(&t.hv, true);
                if sim > 0.55 { Some((cid.clone(), sim)) } else { None }
            }).collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
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
        self.bubble.semantic_memory.insert(
            format!("SPO:{}_{}_{}", subject, predicate, object),
            MemoryTrace::new(triple, MemoryTier::Semantic, 0.0, 0.0, 0.0, None),
        );
    }
    pub fn encode_ast(node: &ASTNode) -> Hypervector {
        match node {
            ASTNode::Leaf { value } => Hypervector::from_string(value, Some(value)),
            ASTNode::Tree { op, left, right } => {
                let op_hv    = Hypervector::from_string(op, Some(op));
                let left_hv  = Self::encode_ast(left);
                let right_hv = Self::encode_ast(right).permute(1);
                op_hv.bind(&bundle(&[left_hv, right_hv]))
            }
        }
    }
    pub fn evaluate_lasso_fitness(candidate: &ASTNode, target: &Hypervector,
                                  lambda_l1: f64) -> f64 {
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
    pub fn learn(&mut self, state: &Hypervector, action_name: &str,
                 reward: f64, _next: Option<&Hypervector>) {
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
