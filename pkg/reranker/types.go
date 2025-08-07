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
			ModelID:     "../../models/jina-reranker-v2-base-multilingual-Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Fast inference", "Multilingual support"},
			Type:        "gguf-local",
		},
		{
			Name:        "mxbai-v1",
			DisplayName: "MixedBread AI Reranker V1",
			Provider:    "MixedBread AI",
			ModelID:     "../../models/mxbai-rerank-large-v2-Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Balanced performance"},
			Type:        "gguf-local",
		},
		{
			Name:        "mxbai-v2",
			DisplayName: "MixedBread AI Reranker V2",
			Provider:    "MixedBread AI", 
			ModelID:     "../../models/mxbai-rerank-large-v2-Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Latest generation", "High accuracy"},
			Type:        "gguf-local",
		},
		{
			Name:        "qwen-0.6b",
			DisplayName: "Qwen Reranker 0.6B",
			Provider:    "Alibaba",
			ModelID:     "../../models/Qwen3-Reranker-0.6B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Fastest", "Smallest model"},
			Type:        "gguf-local",
		},
		{
			Name:        "qwen-4b",
			DisplayName: "Qwen Reranker 4B",
			Provider:    "Alibaba",
			ModelID:     "../../models/Qwen3-Reranker-4B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Balanced size and quality"},
			Type:        "gguf-local",
		},
		{
			Name:        "qwen-8b",
			DisplayName: "Qwen Reranker 8B",
			Provider:    "Alibaba",
			ModelID:     "../../models/Qwen3-Reranker-8B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Largest", "Highest accuracy"},
			Type:        "gguf-local",
		},
		{
			Name:        "ms-marco-v2",
			DisplayName: "MS MARCO MiniLM-L12-v2",
			Provider:    "Microsoft",
			ModelID:     "../../models/ms-marco-MiniLM-L12-v2.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Fast", "Well-established"},
			Type:        "gguf-local",
		},
		{
			Name:        "bge-base",
			DisplayName: "BGE Reranker Base",
			Provider:    "BAAI",
			ModelID:     "../../models/bge-reranker-base-q4_k_m.gguf",
			Strengths:   []string{"Local inference", "Fast", "Lightweight baseline"},
			Type:        "gguf-local",
		},
		{
			Name:        "bge-large",
			DisplayName: "BGE Reranker Large",
			Provider:    "BAAI",
			ModelID:     "../../models/bge-reranker-large-q4_k_m.gguf",
			Strengths:   []string{"Local inference", "Larger", "More accurate"},
			Type:        "gguf-local",
		},
		{
			Name:        "bge-v2-m3",
			DisplayName: "BGE Reranker V2-M3",
			Provider:    "BAAI",
			ModelID:     "../../models/bge-reranker-v2-m3-Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Latest multilingual model"},
			Type:        "gguf-local",
		},
		{
			Name:        "bge-v2-gemma",
			DisplayName: "BGE Reranker V2-Gemma",
			Provider:    "BAAI",
			ModelID:     "../../models/bge-reranker-v2-gemma.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "LLM-based reranker"},
			Type:        "gguf-local",
		},
		{
			Name:        "colbert-v2",
			DisplayName: "ColBERT v2.0",
			Provider:    "Stanford",
			ModelID:     "../../models/colbertv2.0.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "ColBERT architecture", "Efficient retrieval"},
			Type:        "gguf-local",
		},
		{
			Name:        "jina-m0",
			DisplayName: "Jina Reranker M0",
			Provider:    "Jina AI",
			ModelID:     "../../models/jina-reranker-m0-Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Medium size", "Multilingual support"},
			Type:        "gguf-local",
		},
		{
			Name:        "jina-v1-tiny",
			DisplayName: "Jina Reranker V1 Tiny EN",
			Provider:    "Jina AI",
			ModelID:     "../../models/jina-reranker-v1-tiny-en-Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Tiny size", "English only", "Ultra fast"},
			Type:        "gguf-local",
		},
		{
			Name:        "ms-marco-l4-v2",
			DisplayName: "MS MARCO MiniLM-L4-v2",
			Provider:    "Microsoft",
			ModelID:     "../../models/ms-marco-MiniLM-L4-v2.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Ultra fast", "Lightweight", "4-layer model"},
			Type:        "gguf-local",
		},
		// GGUF Local Models
		{
			Name:        "gguf/qwen-0.6b",
			DisplayName: "Qwen Reranker 0.6B (GGUF)",
			Provider:    "Alibaba",
			ModelID:     "../../models/Qwen3-Reranker-0.6B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Fastest", "Smallest model"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/qwen-4b",
			DisplayName: "Qwen Reranker 4B (GGUF)",
			Provider:    "Alibaba",
			ModelID:     "../../models/Qwen3-Reranker-4B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Balanced size and quality"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/qwen-8b",
			DisplayName: "Qwen Reranker 8B (GGUF)",
			Provider:    "Alibaba",
			ModelID:     "../../models/Qwen3-Reranker-8B.Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Largest", "Highest accuracy"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/bge-base",
			DisplayName: "BGE Reranker Base (GGUF)",
			Provider:    "BAAI",
			ModelID:     "../../models/bge-reranker-base-q4_k_m.gguf",
			Strengths:   []string{"Local inference", "Fast", "Lightweight baseline"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/bge-large",
			DisplayName: "BGE Reranker Large (GGUF)",
			Provider:    "BAAI",
			ModelID:     "../../models/bge-reranker-large-q4_k_m.gguf",
			Strengths:   []string{"Local inference", "Larger", "More accurate"},
			Type:        "gguf-local",
		},
		{
			Name:        "gguf/bge-v2-m3",
			DisplayName: "BGE Reranker V2-M3 (GGUF)",
			Provider:    "BAAI",
			ModelID:     "../../models/bge-reranker-v2-m3-Q4_K_M.gguf",
			Strengths:   []string{"Local inference", "Latest multilingual model"},
			Type:        "gguf-local",
		},
	}
}
