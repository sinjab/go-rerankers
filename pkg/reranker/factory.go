package reranker

import (
	"fmt"
)

// RerankerType represents different reranker implementation types
type RerankerType string

const (
	TypeSimple      RerankerType = "simple"
	TypeCrossEncoder RerankerType = "cross-encoder"
	TypeHuggingFace RerankerType = "huggingface"
	TypeHugot       RerankerType = "hugot"
)

// NewReranker creates a new reranker based on the model name and configuration
func NewReranker(config Config) (Reranker, error) {
	// Map model names to implementation types
	modelToType := map[string]RerankerType{
		"simple":                                     TypeSimple,
		
		// Cross-encoder models
		"cross-encoder/ms-marco-MiniLM-L12-v2":      TypeCrossEncoder,
		"BAAI/bge-reranker-base":                     TypeCrossEncoder,
		"BAAI/bge-reranker-large":                    TypeCrossEncoder,
		"BAAI/bge-reranker-v2-m3":                    TypeCrossEncoder,
		"BAAI/bge-reranker-v2-gemma":                 TypeCrossEncoder,
		"BAAI/bge-reranker-v2-minicpm-layerwise":     TypeCrossEncoder,
		"Qwen/Qwen3-Reranker-0.6B":                  TypeCrossEncoder,
		"Qwen/Qwen3-Reranker-4B":                     TypeCrossEncoder,
		"Qwen/Qwen3-Reranker-8B":                     TypeCrossEncoder,
		"mixedbread-ai/mxbai-rerank-large-v1":        TypeCrossEncoder,
		"mixedbread-ai/mxbai-rerank-large-v2":        TypeCrossEncoder,
		"jinaai/jina-reranker-v2-base-multilingual":  TypeCrossEncoder,
		
		// Friendly names mapping to model IDs
		"jina-v2":         TypeCrossEncoder,
		"mxbai-v1":        TypeCrossEncoder,
		"mxbai-v2":        TypeCrossEncoder,
		"qwen-0.6b":       TypeCrossEncoder,
		"qwen-4b":         TypeCrossEncoder,
		"qwen-8b":         TypeCrossEncoder,
		"ms-marco-v2":     TypeCrossEncoder,
		"bge-base":        TypeCrossEncoder,
		"bge-large":       TypeCrossEncoder,
		"bge-v2-m3":       TypeCrossEncoder,
		"bge-v2-gemma":    TypeCrossEncoder,
		"bge-v2-minicpm-layerwise": TypeCrossEncoder,
	}

	// Map friendly names to actual model IDs
	friendlyNameToModelID := map[string]string{
		"jina-v2":         "jinaai/jina-reranker-v2-base-multilingual",
		"mxbai-v1":        "mixedbread-ai/mxbai-rerank-large-v1",
		"mxbai-v2":        "mixedbread-ai/mxbai-rerank-large-v2",
		"qwen-0.6b":       "Qwen/Qwen3-Reranker-0.6B",
		"qwen-4b":         "Qwen/Qwen3-Reranker-4B",
		"qwen-8b":         "Qwen/Qwen3-Reranker-8B",
		"ms-marco-v2":     "cross-encoder/ms-marco-MiniLM-L12-v2",
		"bge-base":        "BAAI/bge-reranker-base",
		"bge-large":       "BAAI/bge-reranker-large",
		"bge-v2-m3":       "BAAI/bge-reranker-v2-m3",
		"bge-v2-gemma":    "BAAI/bge-reranker-v2-gemma",
		"bge-v2-minicpm-layerwise": "BAAI/bge-reranker-v2-minicpm-layerwise",
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
			rerankType = TypeCrossEncoder
		} else {
			// Default to cross-encoder for unknown models
			rerankType = TypeCrossEncoder
		}
	}

	switch rerankType {
	case TypeSimple:
		return NewSimpleReranker(config), nil
		
	case TypeCrossEncoder:
		return NewCrossEncoderReranker(config), nil
		
	case TypeHuggingFace:
		// TODO: Re-enable when dependency issues are resolved
		return nil, fmt.Errorf("%w: HuggingFace reranker temporarily disabled", ErrUnsupportedModel)
		
	case TypeHugot:
		// TODO: Re-enable when dependency issues are resolved
		return nil, fmt.Errorf("%w: Hugot reranker temporarily disabled", ErrUnsupportedModel)
		
	default:
		return nil, fmt.Errorf("%w: unknown reranker type: %s", ErrUnsupportedModel, rerankType)
	}
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
