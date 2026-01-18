use crate::config::LlmConfig;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct OllamaGenerate {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

#[derive(Serialize)]
struct OllamaEmbed {
    model: String,
    input: String,
}

#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

pub struct OllamaClient {
    base_url: String,
    model: String,
    embed_model: String,
    client: reqwest::blocking::Client,
}

impl OllamaClient {
    pub fn new(config: &LlmConfig) -> Self {
        Self {
            base_url: config.endpoint.clone(),
            model: config.model.clone(),
            embed_model: config.embed_model.clone(),
            client: reqwest::blocking::Client::new(),
        }
    }

    pub fn generate(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/api/generate", self.base_url);
        let body = OllamaGenerate {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            stream: false,
        };
        let resp: OllamaGenerateResponse = self.client.post(&url).json(&body).send()?.json()?;
        Ok(resp.response)
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let url = format!("{}/api/embed", self.base_url);
        let body = OllamaEmbed {
            model: self.embed_model.clone(),
            input: text.to_string(),
        };
        let resp: OllamaEmbedResponse = self.client.post(&url).json(&body).send()?.json()?;
        Ok(resp.embeddings.into_iter().next().unwrap_or_default())
    }

    pub fn is_available(&self) -> bool {
        let url = format!("{}/api/tags", self.base_url);
        self.client.get(&url).send().is_ok()
    }
}
