package reranker

import (
	"context"
	"fmt"
)

// Document represents a document to be ranked
type Document struct {
	ID      string                 `json:"id"`
	Content string                 `json:"content"`
	Score   float64                `json:"score"`
	Meta    map[string]interface{} `json:"meta,omitempty"`
}

// TestData represents test data structure
type TestData struct {
	Query       string   `json:"query"`
	Documents   []string `json:"documents"`
	Instruction string   `json:"instruction,omitempty"`
}

// RerankResult represents the result of a reranking operation
type RerankResult struct {
	Document Document `json:"document"`
	Score    float64  `json:"score"`
	Index    int      `json:"index"`
}

// Config holds configuration for rerankers
type Config struct {
	Model     string                 `json:"model"`
	MaxDocs   int                    `json:"max_docs"`
	Threshold float64                `json:"threshold"`
	Device    string                 `json:"device,omitempty"`    // "cpu", "cuda", "auto"
	Options   map[string]interface{} `json:"options,omitempty"`
}

// Reranker interface defines the contract for reranking implementations
type Reranker interface {
	Rerank(ctx context.Context, query string, documents []Document) ([]Document, error)
	ComputeScore(ctx context.Context, query string, documents []Document) ([]float64, error)
	Rank(ctx context.Context, query string, documents []Document, topN int) ([]RerankResult, error)
	Configure(config Config) error
	GetModelName() string
}

// Error types
var (
	ErrModelNotFound     = fmt.Errorf("model not found")
	ErrInvalidInput      = fmt.Errorf("invalid input")
	ErrInitialization    = fmt.Errorf("initialization error")
	ErrInference         = fmt.Errorf("inference error")
	ErrUnsupportedModel  = fmt.Errorf("unsupported model")
)

// ModelInfo represents information about a supported model
type ModelInfo struct {
	Name        string   `json:"name"`
	DisplayName string   `json:"display_name"`
	Provider    string   `json:"provider"`
	ModelID     string   `json:"model_id"`
	Strengths   []string `json:"strengths"`
	Type        string   `json:"type"` // "cross-encoder", "bi-encoder"
}

// GetSupportedModels returns a list of all supported models
func GetSupportedModels() []ModelInfo {
	return []ModelInfo{
		{
			Name:        "jina-v2",
			DisplayName: "Jina Reranker V2",
			Provider:    "Jina AI",
			ModelID:     "jinaai/jina-reranker-v2-base-multilingual",
			Strengths:   []string{"Fast inference", "Multilingual support"},
			Type:        "cross-encoder",
		},
		{
			Name:        "mxbai-v1",
			DisplayName: "MixedBread AI Reranker V1",
			Provider:    "MixedBread AI",
			ModelID:     "mixedbread-ai/mxbai-rerank-large-v1",
			Strengths:   []string{"Balanced performance"},
			Type:        "cross-encoder",
		},
		{
			Name:        "mxbai-v2",
			DisplayName: "MixedBread AI Reranker V2",
			Provider:    "MixedBread AI", 
			ModelID:     "mixedbread-ai/mxbai-rerank-large-v2",
			Strengths:   []string{"Latest generation", "High accuracy"},
			Type:        "cross-encoder",
		},
		{
			Name:        "qwen-0.6b",
			DisplayName: "Qwen Reranker 0.6B",
			Provider:    "Alibaba",
			ModelID:     "Qwen/Qwen3-Reranker-0.6B",
			Strengths:   []string{"Fastest", "Smallest model"},
			Type:        "cross-encoder",
		},
		{
			Name:        "qwen-4b",
			DisplayName: "Qwen Reranker 4B",
			Provider:    "Alibaba",
			ModelID:     "Qwen/Qwen3-Reranker-4B",
			Strengths:   []string{"Balanced size and quality"},
			Type:        "cross-encoder",
		},
		{
			Name:        "qwen-8b",
			DisplayName: "Qwen Reranker 8B",
			Provider:    "Alibaba",
			ModelID:     "Qwen/Qwen3-Reranker-8B",
			Strengths:   []string{"Largest", "Highest accuracy"},
			Type:        "cross-encoder",
		},
		{
			Name:        "ms-marco-v2",
			DisplayName: "MS MARCO MiniLM-L12-v2",
			Provider:    "Microsoft",
			ModelID:     "cross-encoder/ms-marco-MiniLM-L12-v2",
			Strengths:   []string{"Fast", "Well-established"},
			Type:        "cross-encoder",
		},
		{
			Name:        "bge-base",
			DisplayName: "BGE Reranker Base",
			Provider:    "BAAI",
			ModelID:     "BAAI/bge-reranker-base",
			Strengths:   []string{"Fast", "Lightweight baseline"},
			Type:        "cross-encoder",
		},
		{
			Name:        "bge-large",
			DisplayName: "BGE Reranker Large",
			Provider:    "BAAI",
			ModelID:     "BAAI/bge-reranker-large",
			Strengths:   []string{"Larger", "More accurate"},
			Type:        "cross-encoder",
		},
		{
			Name:        "bge-v2-m3",
			DisplayName: "BGE Reranker V2-M3",
			Provider:    "BAAI",
			ModelID:     "BAAI/bge-reranker-v2-m3",
			Strengths:   []string{"Latest multilingual model"},
			Type:        "cross-encoder",
		},
		{
			Name:        "bge-v2-gemma",
			DisplayName: "BGE Reranker V2-Gemma",
			Provider:    "BAAI",
			ModelID:     "BAAI/bge-reranker-v2-gemma",
			Strengths:   []string{"LLM-based reranker"},
			Type:        "cross-encoder",
		},
		{
			Name:        "bge-v2-minicpm-layerwise",
			DisplayName: "BGE Reranker V2-MiniCPM-Layerwise",
			Provider:    "BAAI",
			ModelID:     "BAAI/bge-reranker-v2-minicpm-layerwise",
			Strengths:   []string{"Advanced layerwise model"},
			Type:        "cross-encoder",
		},
		// GGUF Local Models
		{
			Name:        "gguf/qwen-0.6b",
			DisplayName: "Qwen Reranker 0.6B (GGUF)",
			Provider:    "Alibaba",
			ModelID:     "models/Qwen3-Reranker-0.6B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Fastest", "Smallest model"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/qwen-4b",
			DisplayName: "Qwen Reranker 4B (GGUF)",
			Provider:    "Alibaba",
			ModelID:     "models/Qwen3-Reranker-4B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Balanced size and quality"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/qwen-8b",
			DisplayName: "Qwen Reranker 8B (GGUF)",
			Provider:    "Alibaba",
			ModelID:     "models/Qwen3-Reranker-8B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Largest", "Highest accuracy"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/bge-base",
			DisplayName: "BGE Reranker Base (GGUF)",
			Provider:    "BAAI",
			ModelID:     "models/bge-reranker-base-q4_k_m.gguf",
			Strengths:   []string{"Local inference", "Fast", "Lightweight baseline"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/bge-large",
			DisplayName: "BGE Reranker Large (GGUF)",
			Provider:    "BAAI",
			ModelID:     "models/bge-reranker-large-q4_k_m.gguf",
			Strengths:   []string{"Local inference", "Larger", "More accurate"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/bge-v2-m3",
			DisplayName: "BGE Reranker V2-M3 (GGUF)",
			Provider:    "BAAI",
			ModelID:     "models/bge-reranker-v2-m3-Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Latest multilingual model"},
			Type:        "gguf-local",
		},
	}
}
