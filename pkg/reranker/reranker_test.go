package reranker

import (
	"testing"
)

func TestFactoryFunction(t *testing.T) {
	// Test factory with Simple reranker
	config := Config{
		Model:   "simple",
		MaxDocs: 10,
		Device:  "cpu",
	}

	reranker, err := NewReranker(config)
	if err != nil {
		t.Fatalf("NewReranker failed: %v", err)
	}

	if reranker.GetModelName() != "simple" {
		t.Errorf("Expected simple, got %s", reranker.GetModelName())
	}

	// Test factory with Cross-encoder reranker
	config.Model = "mxbai-v2"
	reranker, err = NewReranker(config)
	if err != nil {
		t.Fatalf("NewReranker failed: %v", err)
	}

	if reranker.GetModelName() != "mixedbread-ai/mxbai-rerank-large-v2" {
		t.Errorf("Expected mixedbread-ai/mxbai-rerank-large-v2, got %s", reranker.GetModelName())
	}
}

func TestGetSupportedModels(t *testing.T) {
	models := GetSupportedModels()
	if len(models) == 0 {
		t.Error("Expected at least one supported model")
	}

	// Check that we have the expected models
	expectedModels := []string{"jina-v2", "mxbai-v1", "mxbai-v2", "qwen-0.6b"}
	for _, expected := range expectedModels {
		found := false
		for _, model := range models {
			if model.Name == expected {
				found = true
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
