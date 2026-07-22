#!/usr/bin/env Rscript
# Table S9 — silver-labeling runtime and aggregation settings.
rm(list = ls())
source("_common.R")

OLLAMA_CLIENT_VERSION <- "0.11.4"
OLLAMA_MODEL_DIGEST <- "46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e"
OLLAMA_QUANTIZATION <- "Q4_K_M"
df <- data.frame(
  Setting = c(
    "Base model",
    "Inference engine",
    "Ollama client version",
    "Ollama model digest",
    "Model quantization",
    "Hardware",
    "GPU driver version",
    "Sampling temperature",
    "Max new tokens",
    "Votes per sentence (self-consistency)",
    "Random seed",
    "Vote-adoption rule",
    "Confidence rule",
    "Per-sentence label cap",
    "Sentence segmentation",
    "No-domain fallback"
  ),
  Value = c(
    "Llama-3.1-8B (llama3.1:8b)",
    "Ollama, OpenAI-compatible local endpoint",
    OLLAMA_CLIENT_VERSION,
    OLLAMA_MODEL_DIGEST,
    OLLAMA_QUANTIZATION,
    "4 x NVIDIA A100 80 GB",
    "NVIDIA 535.216.03",
    "0.7",
    "1024",
    "5",
    "42",
    "domain kept if it appears in at least 3 of 5 votes (ratio 0.5)",
    "domain kept if its mean confidence is at least 0.5",
    "3 (requested in the prompt and applied as a hard cap after aggregation)",
    "tokenizer-aligned to the attention model, minimum segment length 30 characters",
    "other_general (C2) when no domain qualifies"
  ),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
write_table(df, "TableS9")
