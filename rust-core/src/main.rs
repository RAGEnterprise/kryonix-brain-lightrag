mod utils;

#[cfg(feature = "axum-server")]
use axum::{
    routing::{get, post},
    Json, Router, extract::State,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "axum-server")]
use std::net::SocketAddr;
#[cfg(feature = "axum-server")]
use std::sync::Arc;
#[cfg(feature = "python-bridge")]
use pyo3::prelude::*;
use tracing::{info};
use crate::utils::SecretScanner;

#[cfg(feature = "axum-server")]
#[derive(Clone)]
struct AppState {}

#[cfg(feature = "axum-server")]
#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    mode: Option<String>,
    lang: Option<String>,
}

#[cfg(feature = "axum-server")]
#[derive(Serialize)]
struct SearchResponse {
    status: String,
    answer: String,
    grounding: serde_json::Value,
}

#[cfg(feature = "axum-server")]
#[derive(Deserialize)]
struct IngestProposeRequest {
    content: String,
    source: String,
    reason: String,
}

#[cfg(feature = "axum-server")]
#[derive(Serialize)]
struct IngestProposeResponse {
    status: String,
    id: String,
    security_alerts: Option<Vec<String>>,
}

#[cfg(feature = "axum-server")]
async fn health() -> &'static str {
    "OK"
}

#[cfg(all(feature = "axum-server", feature = "python-bridge"))]
async fn search(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let query = payload.query;
    let mode = payload.mode.unwrap_or_else(|| "hybrid".to_string());
    let lang = payload.lang.unwrap_or_else(|| "pt-BR".to_string());

    info!("Searching: {} (mode={}, lang={})", query, mode, lang);

    let result = Python::with_gil(|py| -> PyResult<SearchResponse> {
        let _rag_mod = py.import_bound("kryonix_brain_lightrag.rag")?;
        Ok(SearchResponse {
            status: "success".to_string(),
            answer: format!("Resposta (Axum) para: {}", query),
            grounding: serde_json::json!({}),
        })
    });

    match result {
        Ok(res) => Ok(Json(res)),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

#[cfg(feature = "axum-server")]
async fn ingest_propose(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<IngestProposeRequest>,
) -> Json<IngestProposeResponse> {
    let (_sanitized_content, findings) = SecretScanner::scan_and_redact(&payload.content);
    
    let item_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
    
    info!("Ingest proposed: {} from {} (findings: {:?})", item_id, payload.source, findings);
    
    Json(IngestProposeResponse {
        status: "queued".to_string(),
        id: item_id,
        security_alerts: if findings.is_empty() { None } else { Some(findings) },
    })
}

#[cfg(feature = "axum-server")]
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    
    let state = Arc::new(AppState {});

    let mut router = Router::new()
        .route("/health", get(health))
        .route("/ingest/propose", post(ingest_propose));

    #[cfg(feature = "python-bridge")]
    {
        router = router.route("/search", post(search));
    }

    let app = router.with_state(state);

    let port = std::env::var("KRYONIX_BRAIN_PORT")
        .unwrap_or_else(|_| "8000".to_string())
        .parse::<u16>()
        .unwrap_or(8000);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("Kryonix Brain Axum API (Rust) listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[cfg(not(feature = "axum-server"))]
fn main() {
    tracing_subscriber::fmt::init();
    info!("Kryonix Brain Rust Core (Auxiliary Tools Mode)");
    info!("Use --features axum-server to enable the experimental API.");
}
