package reranker

import (
	"context"
	"log"
	"sort"
	"strings"
)

// SimpleReranker implements basic reranking functionality
type SimpleReranker struct {
	config Config
}

// NewSimpleReranker creates a new simple reranker
func NewSimpleReranker(config Config) *SimpleReranker {
	if config.MaxDocs == 0 {
		config.MaxDocs = 100
	}
	if config.Threshold == 0 {
		config.Threshold = 0.0
	}
	
	return &SimpleReranker{
		config: config,
	}
}

// Rerank reorders documents based on relevance to a query
func (r *SimpleReranker) Rerank(ctx context.Context, query string, documents []Document) ([]Document, error) {
	if len(documents) == 0 {
		return documents, nil
	}

	log.Printf("Reranking %d documents for query: %s", len(documents), query)

	// Apply basic text similarity scoring
	for i := range documents {
		documents[i].Score = r.calculateSimilarity(query, documents[i].Content)
	}

	// Sort by score (descending)
	sort.Slice(documents, func(i, j int) bool {
		return documents[i].Score > documents[j].Score
	})

	// Apply threshold filter
	var filtered []Document
	for _, doc := range documents {
		if doc.Score >= r.config.Threshold {
			filtered = append(filtered, doc)
		}
	}

	// Limit to max documents
	if len(filtered) > r.config.MaxDocs {
		filtered = filtered[:r.config.MaxDocs]
	}

	return filtered, nil
}

// Configure updates the reranker configuration
func (r *SimpleReranker) Configure(config Config) error {
	r.config = config
	if r.config.MaxDocs == 0 {
		r.config.MaxDocs = 100
	}
	return nil
}

// calculateSimilarity computes basic text similarity
func (r *SimpleReranker) calculateSimilarity(query, content string) float64 {
	queryWords := strings.Fields(strings.ToLower(query))
	contentWords := strings.Fields(strings.ToLower(content))
	
	if len(queryWords) == 0 || len(contentWords) == 0 {
		return 0.0
	}
	
	matches := 0
	for _, qword := range queryWords {
		for _, cword := range contentWords {
			if strings.Contains(cword, qword) || strings.Contains(qword, cword) {
				matches++
				break
			}
		}
	}
	
	return float64(matches) / float64(len(queryWords))
}

// ComputeScore computes scores for query-document pairs
func (r *SimpleReranker) ComputeScore(ctx context.Context, query string, documents []Document) ([]float64, error) {
	scores := make([]float64, len(documents))
	
	for i, doc := range documents {
		scores[i] = r.calculateSimilarity(query, doc.Content)
	}
	
	return scores, nil
}

// Rank returns top-N ranked documents
func (r *SimpleReranker) Rank(ctx context.Context, query string, documents []Document, topN int) ([]RerankResult, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	// Calculate scores for all documents
	scores, err := r.ComputeScore(ctx, query, documents)
	if err != nil {
		return nil, err
	}

	// Create results with scores and original indices
	results := make([]RerankResult, len(documents))
	for i, doc := range documents {
		results[i] = RerankResult{
			Document: doc,
			Score:    scores[i],
			Index:    i,
		}
	}

	// Sort by score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Apply threshold filter
	var filtered []RerankResult
	for _, result := range results {
		if result.Score >= r.config.Threshold {
			filtered = append(filtered, result)
		}
	}

	// Limit to topN
	if topN > 0 && len(filtered) > topN {
		filtered = filtered[:topN]
	}

	return filtered, nil
}

// GetModelName returns the model name
func (r *SimpleReranker) GetModelName() string {
	if r.config.Model != "" {
		return r.config.Model
	}
	return "simple-reranker"
}
