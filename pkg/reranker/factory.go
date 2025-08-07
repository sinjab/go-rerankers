package reranker

import (
	"fmt"
)

// RerankerType represents different reranker implementation types
type RerankerType string

const (
	TypeGGUFLocal   RerankerType = "gguf-local"
)

// NewReranker creates a new reranker based on the model name and configuration
func NewReranker(config Config) (Reranker, error) {
	// All models use GGUF local inference with real llama.cpp
	modelToType := map[string]RerankerType{
		// All models now use GGUF local inference with real llama.cpp
		"cross-encoder/ms-marco-MiniLM-L12-v2":      TypeGGUFLocal,
		"BAAI/bge-reranker-base":                     TypeGGUFLocal,
		"BAAI/bge-reranker-large":                    TypeGGUFLocal,
		"BAAI/bge-reranker-v2-m3":                    TypeGGUFLocal,
		"BAAI/bge-reranker-v2-gemma":                 TypeGGUFLocal,

		"Qwen/Qwen3-Reranker-0.6B":                  TypeGGUFLocal,
		"Qwen/Qwen3-Reranker-4B":                     TypeGGUFLocal,
		"Qwen/Qwen3-Reranker-8B":                     TypeGGUFLocal,
		"mixedbread-ai/mxbai-rerank-large-v1":        TypeGGUFLocal,
		"mixedbread-ai/mxbai-rerank-large-v2":        TypeGGUFLocal,
		"jinaai/jina-reranker-v2-base-multilingual":  TypeGGUFLocal,
		
		// Additional Jina models
		"jina-m0":                                    TypeGGUFLocal,
		"jina-v1-tiny":                               TypeGGUFLocal,
		"ms-marco-l4-v2":                             TypeGGUFLocal,
		
		// GGUF local models (explicit GGUF paths)
		"gguf/qwen-0.6b":       TypeGGUFLocal,
		"gguf/qwen-4b":         TypeGGUFLocal,
		"gguf/qwen-8b":         TypeGGUFLocal,
		"gguf/bge-base":        TypeGGUFLocal,
		"gguf/bge-large":       TypeGGUFLocal,
		"gguf/bge-v2-m3":       TypeGGUFLocal,
		
		// Friendly names mapping to GGUF models
		"jina-v2":         TypeGGUFLocal,
		"mxbai-v1":        TypeGGUFLocal,
		"mxbai-v2":        TypeGGUFLocal,
		"qwen-0.6b":       TypeGGUFLocal,
		"qwen-4b":         TypeGGUFLocal,
		"qwen-8b":         TypeGGUFLocal,
		"ms-marco-v2":     TypeGGUFLocal,
		"bge-base":        TypeGGUFLocal,
		"bge-large":       TypeGGUFLocal,
		"bge-v2-m3":       TypeGGUFLocal,
		"bge-v2-gemma":    TypeGGUFLocal,
		"colbert-v2":               TypeGGUFLocal,
	}

	// Map friendly names to GGUF model files - all models now use real llama.cpp inference
	friendlyNameToModelID := map[string]string{
		// Friendly names now point directly to GGUF files
		"jina-v2":         "models/jina-reranker-v2-base-multilingual-Q4_K_M.gguf",
		"mxbai-v1":        "models/mxbai-rerank-large-v2-Q4_K_M.gguf", // Use v2 for v1 as well
		"mxbai-v2":        "models/mxbai-rerank-large-v2-Q4_K_M.gguf",
		"qwen-0.6b":       "models/Qwen3-Reranker-0.6B.Q4_K_M.gguf",
		"qwen-4b":         "models/Qwen3-Reranker-4B.Q4_K_M.gguf",
		"qwen-8b":         "models/Qwen3-Reranker-8B.Q4_K_M.gguf",
		"ms-marco-v2":     "models/ms-marco-MiniLM-L12-v2.Q4_K_M.gguf",
		"bge-base":        "models/bge-reranker-base-q4_k_m.gguf",
		"bge-large":       "models/bge-reranker-large-q4_k_m.gguf",
		"bge-v2-m3":       "models/bge-reranker-v2-m3-Q4_K_M.gguf",
		"bge-v2-gemma":    "models/bge-reranker-v2-gemma.Q4_K_M.gguf",
		"colbert-v2":               "models/colbertv2.0.Q4_K_M.gguf",
		"jina-m0":                  "models/jina-reranker-m0-Q4_K_M.gguf",
		"jina-v1-tiny":             "models/jina-reranker-v1-tiny-en-Q4_K_M.gguf",
		"ms-marco-l4-v2":           "models/ms-marco-MiniLM-L4-v2.Q4_K_M.gguf",
		
		// Full model IDs also point to GGUF files
		"jinaai/jina-reranker-v2-base-multilingual":  "models/jina-reranker-v2-base-multilingual-Q4_K_M.gguf",
		"mixedbread-ai/mxbai-rerank-large-v1":        "models/mxbai-rerank-large-v2-Q4_K_M.gguf",
		"mixedbread-ai/mxbai-rerank-large-v2":        "models/mxbai-rerank-large-v2-Q4_K_M.gguf",
		"Qwen/Qwen3-Reranker-0.6B":                  "models/Qwen3-Reranker-0.6B.Q4_K_M.gguf",
		"Qwen/Qwen3-Reranker-4B":                     "models/Qwen3-Reranker-4B.Q4_K_M.gguf",
		"Qwen/Qwen3-Reranker-8B":                     "models/Qwen3-Reranker-8B.Q4_K_M.gguf",
		"cross-encoder/ms-marco-MiniLM-L12-v2":      "models/ms-marco-MiniLM-L12-v2.Q4_K_M.gguf",
		"BAAI/bge-reranker-base":                     "models/bge-reranker-base-q4_k_m.gguf",
		"BAAI/bge-reranker-large":                    "models/bge-reranker-large-q4_k_m.gguf",
		"BAAI/bge-reranker-v2-m3":                    "models/bge-reranker-v2-m3-Q4_K_M.gguf",
		"BAAI/bge-reranker-v2-gemma":                 "models/bge-reranker-v2-gemma.Q4_K_M.gguf",

		
		// GGUF model paths (explicit GGUF paths)
		"gguf/qwen-0.6b":  "models/Qwen3-Reranker-0.6B.Q4_K_M.gguf",
		"gguf/qwen-4b":    "models/Qwen3-Reranker-4B.Q4_K_M.gguf",
		"gguf/qwen-8b":    "models/Qwen3-Reranker-8B.Q4_K_M.gguf",
		"gguf/bge-base":   "models/bge-reranker-base-q4_k_m.gguf",
		"gguf/bge-large":  "models/bge-reranker-large-q4_k_m.gguf",
		"gguf/bge-v2-m3":  "models/bge-reranker-v2-m3-Q4_K_M.gguf",
	}

	// If using a friendly name, convert to model ID
	if modelID, exists := friendlyNameToModelID[config.Model]; exists {
		config.Model = modelID
	}

	// Set defaults
	if config.MaxDocs == 0 {
		config.MaxDocs = 100
	}
	if config.Device == "" {
		config.Device = "auto"
	}

	rerankType, exists := modelToType[config.Model]
	if !exists {
		// Check if it's a friendly name we haven't mapped
		originalModel := config.Model
		if modelID, friendlyExists := friendlyNameToModelID[originalModel]; friendlyExists {
			config.Model = modelID
			rerankType = TypeGGUFLocal
		} else {
			// Default to GGUF local for unknown models (all models now use real inference)
			rerankType = TypeGGUFLocal
		}
	}

	// Only GGUF local inference is supported
	if rerankType != TypeGGUFLocal {
		return nil, fmt.Errorf("%w: only GGUF local inference is supported, got: %s", ErrUnsupportedModel, rerankType)
	}
	
	return NewGGUFLocalReranker(config)
}

// GetAvailableModels returns a list of all available model names
func GetAvailableModels() []string {
	models := GetSupportedModels()
	names := make([]string, len(models))
	for i, model := range models {
		names[i] = model.Name
	}
	return names
}

// GetModelByName returns model info by name
func GetModelByName(name string) (*ModelInfo, error) {
	models := GetSupportedModels()
	for _, model := range models {
		if model.Name == name {
			return &model, nil
		}
	}
	return nil, fmt.Errorf("%w: model %s not found", ErrModelNotFound, name)
}
