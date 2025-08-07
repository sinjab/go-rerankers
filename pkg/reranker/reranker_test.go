package reranker

import (
	"strings"
	"testing"
)

func TestFactoryFunction(t *testing.T) {
	// Test factory with GGUF local reranker - all models now use GGUF
	config := Config{
		Model:   "mxbai-v2",
		MaxDocs: 10,
		Device:  "cpu",
	}

	reranker, err := NewReranker(config)
	if err != nil {
		t.Skipf("Skipping GGUF test due to initialization error: %v", err)
	}
	defer func() {
		if closer, ok := reranker.(interface{ Close() error }); ok {
			closer.Close()
		}
	}()

	// All models should now return GGUF model paths
	modelName := reranker.GetModelName()
	if !strings.Contains(modelName, "models/") {
		t.Errorf("Expected GGUF model path containing 'models/', got %s", modelName)
	}
}

func TestGetSupportedModels(t *testing.T) {
	models := GetSupportedModels()
	if len(models) == 0 {
		t.Error("Expected at least one supported model")
	}

	// Check that we have all expected GGUF models
	expectedModels := []string{
		"jina-v2", "jina-m0", "jina-v1-tiny",
		"mxbai-v1", "mxbai-v2",
		"qwen-0.6b", "qwen-4b", "qwen-8b",
		"ms-marco-v2", "ms-marco-l4-v2",
		"bge-base", "bge-large", "bge-v2-m3", "bge-v2-gemma", "colbert-v2",
	}
	for _, expected := range expectedModels {
		found := false
		for _, model := range models {
			if model.Name == expected {
				found = true
				// Verify all models are GGUF local type
				if model.Type != "gguf-local" {
					t.Errorf("Expected model %s to be gguf-local type, got %s", expected, model.Type)
				}
				break
			}
		}
		if !found {
			t.Errorf("Expected model %s not found in supported models", expected)
		}
	}
}

func TestGetModelByName(t *testing.T) {
	// Test existing model
	model, err := GetModelByName("mxbai-v2")
	if err != nil {
		t.Fatalf("GetModelByName failed: %v", err)
	}

	if model.Name != "mxbai-v2" {
		t.Errorf("Expected name mxbai-v2, got %s", model.Name)
	}

	// Test non-existing model
	_, err = GetModelByName("non-existent-model")
	if err == nil {
		t.Error("Expected error for non-existent model")
	}
}

// TestAllGGUFModelsInitialization tests that all GGUF models can be initialized
func TestAllGGUFModelsInitialization(t *testing.T) {
	// Get all supported models
	models := GetSupportedModels()
	
	// Test each model initialization
	for _, model := range models {
		t.Run(model.Name, func(t *testing.T) {
			config := Config{
				Model:   model.Name,
				MaxDocs: 5, // Small number for testing
				Device:  "cpu",
			}
			
			reranker, err := NewReranker(config)
			if err != nil {
				t.Skipf("Skipping %s due to initialization error: %v", model.Name, err)
				return
			}
			
			// Ensure proper cleanup
			defer func() {
				if closer, ok := reranker.(interface{ Close() error }); ok {
					closer.Close()
				}
			}()
			
			// Verify model name contains the expected GGUF path
			modelName := reranker.GetModelName()
			if !strings.Contains(modelName, "models/") || !strings.Contains(modelName, ".gguf") {
				t.Errorf("Expected GGUF model path for %s, got %s", model.Name, modelName)
			}
			
			t.Logf("Successfully initialized %s -> %s", model.Name, modelName)
		})
	}
}
