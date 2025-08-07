package reranker

import (
	"context"
	"testing"
)

func TestSimpleReranker(t *testing.T) {
	config := Config{
		Model:     "simple",
		MaxDocs:   10,
		Threshold: 0.0,
		Device:    "cpu",
	}

	reranker := NewSimpleReranker(config)
	if reranker == nil {
		t.Fatal("Expected reranker to be created")
	}

	if reranker.GetModelName() != "simple" {
		t.Errorf("Expected model name 'simple', got %s", reranker.GetModelName())
	}

	documents := []Document{
		{ID: "1", Content: "Machine learning is a powerful technology"},
		{ID: "2", Content: "Cooking is an art form"},
		{ID: "3", Content: "Artificial intelligence and machine learning are related"},
	}

	query := "machine learning"
	ctx := context.Background()

	// Test Rerank
	results, err := reranker.Rerank(ctx, query, documents)
	if err != nil {
		t.Fatalf("Rerank failed: %v", err)
	}

	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}

	// Check that results are sorted by score (descending)
	for i := 1; i < len(results); i++ {
		if results[i-1].Score < results[i].Score {
			t.Errorf("Results not properly sorted by score")
		}
	}

	// Test ComputeScore
	scores, err := reranker.ComputeScore(ctx, query, documents)
	if err != nil {
		t.Fatalf("ComputeScore failed: %v", err)
	}

	if len(scores) != len(documents) {
		t.Errorf("Expected %d scores, got %d", len(documents), len(scores))
	}

	// Test Rank with topN
	rankResults, err := reranker.Rank(ctx, query, documents, 2)
	if err != nil {
		t.Fatalf("Rank failed: %v", err)
	}

	if len(rankResults) != 2 {
		t.Errorf("Expected 2 rank results, got %d", len(rankResults))
	}
}