package main

import (
	"context"
	"fmt"
	"log"

	"go-rerankers/pkg/reranker"
)

func main() {
	// Create sample documents
	documents := []reranker.Document{
		{
			ID:      "1",
			Content: "Machine learning algorithms for text classification",
			Score:   0.0,
		},
		{
			ID:      "2",
			Content: "Deep learning neural networks and computer vision",
			Score:   0.0,
		},
		{
			ID:      "3",
			Content: "Natural language processing and sentiment analysis",
			Score:   0.0,
		},
		{
			ID:      "4",
			Content: "Database optimization and performance tuning",
			Score:   0.0,
		},
	}

	// Configure reranker
	config := reranker.Config{
		Model:     "simple",
		MaxDocs:   10,
		Threshold: 0.1,
	}

	// Create reranker
	r := reranker.NewSimpleReranker(config)

	// Test query
	query := "machine learning algorithms"
	
	fmt.Printf("Original documents:\n")
	for i, doc := range documents {
		fmt.Printf("%d. [%s] %s (score: %.2f)\n", i+1, doc.ID, doc.Content, doc.Score)
	}

	// Rerank documents
	reranked, err := r.Rerank(context.Background(), query, documents)
	if err != nil {
		log.Fatalf("Error reranking: %v", err)
	}

	fmt.Printf("\nReranked documents for query '%s':\n", query)
	for i, doc := range reranked {
		fmt.Printf("%d. [%s] %s (score: %.2f)\n", i+1, doc.ID, doc.Content, doc.Score)
	}
}
