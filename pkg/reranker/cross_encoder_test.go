package reranker

import (
	"context"
	"testing"
)

func TestCrossEncoderReranker(t *testing.T) {
	config := Config{
		Model:   "cross-encoder/ms-marco-MiniLM-L12-v2",
		MaxDocs: 10,
	}
	
	reranker := NewCrossEncoderReranker(config)
	
	documents := []Document{
		{
			ID:      "1",
			Content: "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
			Score:   0.0,
		},
		{
			ID:      "2",
			Content: "Berlin is well known for its museums.",
			Score:   0.0,
		},
		{
			ID:      "3",
			Content: "New York City is famous for the Metropolitan Museum of Art.",
			Score:   0.0,
		},
	}
	
	reranked, err := reranker.Rerank(context.Background(), "How many people live in Berlin?", documents)
	if err != nil {
		t.Fatalf("Rerank() returned error: %v", err)
	}
	
	if len(reranked) == 0 {
		t.Error("Expected reranked documents, got none")
	}
	
	// Check that documents are sorted by score (descending)
	for i := 1; i < len(reranked); i++ {
		if reranked[i].Score > reranked[i-1].Score {
			t.Error("Documents are not sorted by score in descending order")
		}
	}
}

func TestCrossEncoderRerankerEmptyDocuments(t *testing.T) {
	config := Config{
		Model: "cross-encoder/ms-marco-MiniLM-L12-v2",
	}
	
	reranker := NewCrossEncoderReranker(config)
	
	var documents []Document
	
	reranked, err := reranker.Rerank(context.Background(), "test query", documents)
	if err != nil {
		t.Fatalf("Rerank() returned error: %v", err)
	}
	
	if len(reranked) != 0 {
		t.Errorf("Expected no documents, got %d", len(reranked))
	}
}

func TestCrossEncoderRerankerConfigure(t *testing.T) {
	config := Config{
		Model: "cross-encoder/ms-marco-MiniLM-L12-v2",
	}
	
	reranker := NewCrossEncoderReranker(config)
	
	newConfig := Config{
		Model:     "cross-encoder/ms-marco-MiniLM-L12-v2",
		MaxDocs:   5,
		Threshold: 0.5,
	}
	
	err := reranker.Configure(newConfig)
	if err != nil {
		t.Fatalf("Configure() returned error: %v", err)
	}
}
