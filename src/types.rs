use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Tool {
    pub name: String,
    pub path: String,
    pub description: String,
    pub synopsis: String,
    pub examples: String,
    pub flags: String,
    pub source: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub commands: Vec<String>,
    pub explanation: String,
    pub warnings: Vec<String>,
    pub needs_confirmation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RiskLevel {
    Safe,
    Review,
    Dangerous,
    Blocked,
}

#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub query: String,
    pub commands: Vec<String>,
    pub executed: bool,
    pub succeeded: bool,
}
