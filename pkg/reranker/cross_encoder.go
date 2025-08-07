package reranker

import (
	"context"
	"log"
	"sort"
	"strings"
	"time"
)

// CrossEncoderReranker implements reranking using a cross-encoder model
type CrossEncoderReranker struct {
	config    Config
	modelPath string
}

// NewCrossEncoderReranker creates a new cross-encoder reranker
func NewCrossEncoderReranker(config Config) *CrossEncoderReranker {
	if config.Model == "" {
		config.Model = "cross-encoder/ms-marco-MiniLM-L12-v2"
	}
	
	if config.MaxDocs == 0 {
		config.MaxDocs = 100
	}
	
	return &CrossEncoderReranker{
		config:    config,
		modelPath: config.Model,
	}
}

// Supported models
const (
	ModelMSMARCO  = "cross-encoder/ms-marco-MiniLM-L12-v2"
	ModelBGERerankerLarge = "BAAI/bge-reranker-large"
	ModelBGERerankerBase  = "BAAI/bge-reranker-base"
	ModelBGERerankerV2M3  = "BAAI/bge-reranker-v2-m3"
	ModelBGERerankerV2Gemma = "BAAI/bge-reranker-v2-gemma"
	ModelBGERerankerV2MiniCPMLayerwise = "BAAI/bge-reranker-v2-minicpm-layerwise"
	ModelQwen3Reranker06B = "Qwen/Qwen3-Reranker-0.6B"
	ModelQwen3Reranker4B = "Qwen/Qwen3-Reranker-4B"
	ModelQwen3Reranker8B = "Qwen/Qwen3-Reranker-8B"
	ModelMxbaiRerankLargeV1 = "mixedbread-ai/mxbai-rerank-large-v1"
	ModelMxbaiRerankLargeV2 = "mixedbread-ai/mxbai-rerank-large-v2"
	ModelJinaRerankerV2BaseMultilingual = "jinaai/jina-reranker-v2-base-multilingual"
)

// Rerank reorders documents based on relevance to a query using cross-encoder scoring
func (r *CrossEncoderReranker) Rerank(ctx context.Context, query string, documents []Document) ([]Document, error) {
	if len(documents) == 0 {
		return documents, nil
	}

	log.Printf("Reranking %d documents for query: %s using cross-encoder model: %s", len(documents), query, r.modelPath)

	// For each document, create a pair with the query for scoring
	pairs := make([][2]string, len(documents))
	for i, doc := range documents {
		pairs[i] = [2]string{query, doc.Content}
	}

	// Calculate scores using cross-encoder logic
	// In a real implementation, this would call a model service
	scores := r.calculateScores(pairs)

	// Apply scores to documents
	for i := range documents {
		documents[i].Score = scores[i]
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

// calculateScores computes scores for query-document pairs
// This is a simplified implementation - in practice, this would call a model service
func (r *CrossEncoderReranker) calculateScores(pairs [][2]string) []float64 {
	scores := make([]float64, len(pairs))
	
	// This is a placeholder implementation that simulates cross-encoder scoring
	// In a real implementation, this would call a model API or local model
	for i, pair := range pairs {
		query := strings.ToLower(pair[0])
		content := strings.ToLower(pair[1])
		
		// Simple word matching algorithm to simulate cross-encoder behavior
		queryWords := strings.Fields(query)
		contentWords := strings.Fields(content)
		
		if len(queryWords) == 0 || len(contentWords) == 0 {
			scores[i] = -5.0 // Default low score for empty content
			continue
		}
		
		// Count matching words with partial matching
		matches := 0
		totalQueryWords := 0
		
		for _, qword := range queryWords {
			// Skip very short words that are likely stop words
			if len(qword) < 2 {
				continue
			}
			totalQueryWords++
			
			for _, cword := range contentWords {
				// Check for exact matches or partial matches
				if qword == cword || strings.Contains(cword, qword) || strings.Contains(qword, cword) {
					matches++
					break
				}
			}
		}
		
		// Avoid division by zero
		if totalQueryWords == 0 {
			scores[i] = -5.0
			continue
		}
		
		// Calculate similarity score (0.0 to 1.0)
		similarity := float64(matches) / float64(totalQueryWords)
		
		// Convert to cross-encoder-like score range based on model
		switch r.modelPath {
		case ModelBGERerankerLarge, ModelBGERerankerBase, ModelBGERerankerV2M3, ModelBGERerankerV2Gemma, ModelBGERerankerV2MiniCPMLayerwise, ModelQwen3Reranker06B, ModelQwen3Reranker4B, ModelQwen3Reranker8B, ModelMxbaiRerankLargeV1, ModelMxbaiRerankLargeV2, ModelJinaRerankerV2BaseMultilingual:
			// BGE reranker models typically output unbounded scores
			// Qwen3 reranker models also use similar range
			// Mxbai reranker models also use similar range
			// Jina AI reranker models also use similar range
			scores[i] = similarity * 20.0 - 10.0
		default:
			// Default cross-encoder model range
			scores[i] = similarity * 15.0 - 5.0
		}
	}
	
	return scores
}

// ComputeScore computes scores for query-document pairs
func (r *CrossEncoderReranker) ComputeScore(ctx context.Context, query string, documents []Document) ([]float64, error) {
	if len(documents) == 0 {
		return nil, nil
	}

	// Create pairs for scoring
	pairs := make([][2]string, len(documents))
	for i, doc := range documents {
		pairs[i] = [2]string{query, doc.Content}
	}

	// Calculate scores using cross-encoder logic
	return r.calculateScores(pairs), nil
}

// Rank returns top-N ranked documents
func (r *CrossEncoderReranker) Rank(ctx context.Context, query string, documents []Document, topN int) ([]RerankResult, error) {
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
func (r *CrossEncoderReranker) GetModelName() string {
	return r.config.Model
}

// Configure updates the reranker configuration
func (r *CrossEncoderReranker) Configure(config Config) error {
	r.config = config
	if r.config.MaxDocs == 0 {
		r.config.MaxDocs = 100
	}
	return nil
}

// CrossEncoderRequest represents the request structure for cross-encoder API
type CrossEncoderRequest struct {
	Model string     `json:"model"`
	Pairs [][2]string `json:"pairs"`
}

// CrossEncoderResponse represents the response structure from cross-encoder API
type CrossEncoderResponse struct {
	Scores []float64 `json:"scores"`
}

// callCrossEncoderAPI would call a real cross-encoder API in production
func (r *CrossEncoderReranker) callCrossEncoderAPI(ctx context.Context, pairs [][2]string) ([]float64, error) {
	// This is a placeholder for actual API call
	// In production, this would make an HTTP request to a model service
	
	// Simulate network delay
	time.Sleep(100 * time.Millisecond)
	
	// Return simulated scores
	return r.calculateScores(pairs), nil
}
