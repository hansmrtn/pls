use crate::db::load_all_tools;
use crate::ollama::OllamaClient;
use crate::types::Tool;

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

pub fn retrieve_relevant_tools(
    client: &OllamaClient,
    conn: &rusqlite::Connection,
    query: &str,
    top_k: usize,
) -> Result<Vec<Tool>, Box<dyn std::error::Error>> {
    let query_embedding = client.embed(query)?;
    let all_tools = load_all_tools(conn)?;

    let mut scored: Vec<(f32, Tool)> = all_tools
        .into_iter()
        .map(|tool| (cosine_similarity(&query_embedding, &tool.embedding), tool))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored.into_iter().take(top_k).map(|(_, t)| t).collect())
}
