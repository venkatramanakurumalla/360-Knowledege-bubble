use serde_json::json;

/// A simple bridge to a local Ollama instance for generating AI responses.
pub struct TinyAiGenerator;

impl TinyAiGenerator {
    pub fn new() -> Self {
        Self {}
    }

    /// Synthesizes an answer using the local Ollama API.
    pub fn synthesize_answer(&self, context: &str, query: &str) -> String {
        // Validate inputs
        if context.trim().is_empty() {
            return "Error: No context provided from memory.".to_string();
        }
        if query.trim().is_empty() {
            return "Error: No query provided.".to_string();
        }

        // Truncate context if too long (Ollama has context limits)
        let max_context_len = 2000;
        let truncated_context = if context.len() > max_context_len {
            format!("{}... [truncated]", &context[..max_context_len])
        } else {
            context.to_string()
        };

        // Format the user message cleanly
        let user_message = format!(
            "Context from database: {}\n\nStudent's Question: {}",
            truncated_context, query
        );

        // Create HTTP client with timeout
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .connect_timeout(std::time::Duration::from_secs(10))
            .build() 
        {
            Ok(c) => c,
            Err(e) => return format!("❌ Failed to create HTTP client: {}", e),
        };
        
        // SWITCHED: Now using /api/chat instead of /api/generate
        let res = client.post("http://localhost:11434/api/chat")
            .json(&json!({
                "model": "qwen2.5:0.5b", // <--- UPGRADED TO QWEN
                "messages": [
                    {
                        "role": "system",
                        // UPGRADED: Strict prompt to prevent rambling and forbid LaTeX math
                        "content": "You are a direct, expert physics and science tutor. Answer the student's question immediately using ONLY the provided context. Do NOT use conversational filler like 'Certainly' or 'Based on the context'. NEVER use LaTeX formatting (like $$ or \\frac). Use plain text and standard keyboard symbols for all math equations (for example, write F = (M/m) * F_net)."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "stream": false,
                "options": {
                    "temperature": 0.1, // <--- LOWERED so it stays highly factual
                    "num_predict": 500
                }
            }))
            .send();

        match res {
            Ok(response) => {
                match response.json::<serde_json::Value>() {
                    Ok(json_data) => {
                        // Extracting from ["message"]["content"]
                        if let Some(text) = json_data["message"]["content"].as_str() {
                            let cleaned = text.trim();
                            if cleaned.is_empty() {
                                "⚠️ AI returned empty response.".to_string()
                            } else {
                                // Updated the footer message
                                format!("🤖 {}\n\n[Powered by Qwen 2.5 (0.5B) via Ollama]", cleaned)
                            }
                        } else {
                            format!("⚠️ Unexpected response format: {:?}", json_data)
                        }
                    }
                    Err(e) => format!("❌ Failed to parse JSON: {}", e),
                }
            }
            Err(e) => {
                if e.is_connect() {
                    // Updated the error instructions
                    "❌ Cannot connect to Ollama.\n\nPlease ensure:\n1. Ollama is installed: https://ollama.com\n2. Ollama is running (run 'ollama serve' in terminal)\n3. Qwen model is pulled: 'ollama pull qwen2.5:0.5b'".to_string()
                } else if e.is_timeout() {
                    "⏱️ Request timed out. Ollama may be loading the model.".to_string()
                } else {
                    format!("❌ Error connecting to Ollama: {}", e)
                }
            }
        }
    }

    /// Quick check if Ollama is available
    pub fn is_available(&self) -> bool {
        if let Ok(client) = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build() 
        {
            client.get("http://localhost:11434/api/tags")
                .send()
                .map(|r| r.status().is_success())
                .unwrap_or(false)
        } else {
            false
        }
    }
}

