use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::collections::HashSet;
use std::time::Duration;


#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Settings {
    pub server: ServerConfig,
    pub inference: InferenceConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct InferenceConfig {
    // Common fields all providers need
    pub base_url: String,
    #[serde(default = "default_model")]
    pub default_model: String,
    #[serde(default)]
    pub allowed_models: Option<HashSet<String>>, // Optional list of allowed models
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub http: Option<HttpConfigSchema>,
    // Provider-specific configuration
    #[serde(flatten)]
    pub provider: InferenceProvider,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "provider", rename_all = "lowercase")]
pub enum InferenceProvider {
    #[serde(rename = "lmstudio")]
    LMStudio,  // No extra fields needed
    
    #[serde(rename = "triton")]
    Triton {
        model_version: String,
    },
    
    #[serde(rename = "openai")]
    OpenAI {
        api_key: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        organization_id: Option<String>,
    },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_format")]
    pub format: LogFormat,
    #[serde(default = "default_log_output")]
    pub output: LogOutput,
    pub file: Option<FileLoggingConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    Json,
    Pretty,
    Compact,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum LogOutput {
    Stdout,
    File,
    Both,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FileLoggingConfig {
    #[serde(default = "default_log_directory")]
    pub directory: PathBuf,
    #[serde(default = "default_log_file_prefix")]
    pub prefix: String,
    #[serde(default = "default_log_file_max_size_mb")]
    pub max_file_size_mb: u64,
    #[serde(default = "default_log_file_max_files")]
    pub max_files: u32,
    #[serde(default = "default_rotation_policy")]
    pub rotation_policy: RotationPolicy,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum RotationPolicy {
    Daily,
    Hourly,
    Size,
}

/// Configuration as it appears in YAML/env
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
pub struct HttpConfigSchema {
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_connect_timeout_secs")]
    pub connect_timeout_secs: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_retry_backoff_ms")]
    pub retry_backoff_ms: u64,
    #[serde(default = "default_keep_alive_secs")]
    pub keep_alive_secs: Option<u64>,
    #[serde(default = "default_max_idle_connections")]
    pub max_idle_connections: Option<usize>,
}

fn default_connect_timeout_secs() -> u64 {
    30
}

fn default_max_retries() -> u32 {
    3
}

fn default_retry_backoff_ms() -> u64 {
    250
}

fn default_keep_alive_secs() -> Option<u64> {
    Some(60)
}

fn default_max_idle_connections() -> Option<usize> {
    Some(10)
}

fn default_model() -> String {
    "gpt-oss-20b".to_string()
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}

fn default_port() -> u16 {
    3000
}

fn default_timeout_secs() -> u64 {
    60
}

fn default_log_output() -> LogOutput {
    LogOutput::Stdout
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_log_format() -> LogFormat {
    LogFormat::Pretty
}

fn default_log_directory() -> PathBuf {
    PathBuf::from("./logs")
}

fn default_log_file_prefix() -> String {
    "app".to_string()
}

fn default_log_file_max_size_mb() -> u64 {
    10
}

fn default_log_file_max_files() -> u32 {
    10
}

fn default_rotation_policy() -> RotationPolicy {
    RotationPolicy::Daily
}


impl HttpConfigSchema {
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_secs)
    }
    
    pub fn connect_timeout(&self) -> Duration {
        Duration::from_secs(self.connect_timeout_secs)
    }
    
    pub fn keep_alive(&self) -> Option<Duration> {
        self.keep_alive_secs.map(Duration::from_secs)
    }
}

impl InferenceConfig {
    // pick back up when we add providers
    #[allow(dead_code)]
    pub fn provider_name(&self) -> &str {
        match &self.provider {
            InferenceProvider::LMStudio => "lmstudio",
            InferenceProvider::Triton { .. } => "triton",
            InferenceProvider::OpenAI { .. } => "openai",
        }
    }
    #[allow(dead_code)]
    pub fn requires_auth(&self) -> bool {
        matches!(self.provider, InferenceProvider::OpenAI { .. })
    }
}

impl Settings {
    pub fn new() -> Result<Self, config::ConfigError> {
        let config = config::Config::builder()
        // Try both .yaml and .yml extensions
        .add_source(
            config::File::with_name("config/default")
                .required(false)
        )
        
        // Show what environment we're trying
        .add_source(
            config::File::with_name(&format!("config/{}", 
                std::env::var("RUN_ENV").unwrap_or_else(|_| {
                    println!("RUN_ENV not set, using 'development'");
                    "development".to_string()
                })
            ))
            .required(false)
        )
        
        .add_source(
            config::Environment::with_prefix("INFERENCE")
                .separator("_")
                .try_parsing(true)
        )
        
        .build()?;
    
    config.try_deserialize()
    }
}