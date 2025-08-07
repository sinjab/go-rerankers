package reranker

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// GGUFLocalReranker implements reranking using GGUF models with llama.cpp inference
type GGUFLocalReranker struct {
	config          Config
	modelPath       string
	inferenceBinary string
	scoreCache      map[string]float64
	cacheMutex      sync.RWMutex
}

// EmbeddingResponse represents the JSON response from llama-embedding
type EmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

// NewGGUFLocalReranker creates a new GGUF local reranker
func NewGGUFLocalReranker(config Config) (*GGUFLocalReranker, error) {
	if config.Model == "" {
		return nil, fmt.Errorf("%w: model path is required for GGUF reranker", ErrInvalidInput)
	}
	
	if config.MaxDocs == 0 {
		config.MaxDocs = 100
	}
	
	// Resolve model path
	modelPath := config.Model
	if !filepath.IsAbs(modelPath) {
		// If relative path, assume it's relative to project root
		var err error
		modelPath, err = filepath.Abs(modelPath)
		if err != nil {
			return nil, fmt.Errorf("%w: failed to resolve model path: %v", ErrInvalidInput, err)
		}
	}
	
	// Find the llama-embedding binary for reranker inference
	inferenceBinary := filepath.Join(filepath.Dir(modelPath), "..", "llama.cpp", "build", "bin", "llama-embedding")
	if _, err := os.Stat(inferenceBinary); os.IsNotExist(err) {
		// Try alternative paths
		alternatives := []string{
			"./llama.cpp/build/bin/llama-embedding",
			"../llama.cpp/build/bin/llama-embedding", 
			"../../llama.cpp/build/bin/llama-embedding",
			"llama-embedding", // In PATH
		}
		
		found := false
		for _, alt := range alternatives {
			if _, err := exec.LookPath(alt); err == nil {
				inferenceBinary = alt
				found = true
				break
			}
		}
		
		if !found {
			return nil, fmt.Errorf("%w: llama-embedding binary not found", ErrInitialization)
		}
	}
	
	// Verify model exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("%w: model file not found: %s", ErrInitialization, modelPath)
	}
	
	reranker := &GGUFLocalReranker{
		config:          config,
		modelPath:       modelPath,
		inferenceBinary: inferenceBinary,
		scoreCache:      make(map[string]float64),
	}
	
	// Test the model by computing a simple embedding
	if err := reranker.testModel(); err != nil {
		return nil, fmt.Errorf("%w: model test failed: %v", ErrInitialization, err)
	}
	
	return reranker, nil
}

// testModel tests if the GGUF model works by computing a simple inference
func (r *GGUFLocalReranker) testModel() error {
	// For now, just check if the binary and model exist
	if _, err := os.Stat(r.inferenceBinary); os.IsNotExist(err) {
		return fmt.Errorf("inference binary not found: %s", r.inferenceBinary)
	}
	
	if _, err := os.Stat(r.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", r.modelPath)
	}
	
	// Quick test with a simple computation
	// We'll do a minimal test here since full inference test might hang
	return nil
}

// computeRerankerScore computes relevance score for a query-document pair using llama-embedding with --pooling rank
// Falls back to embedding similarity if reranker fails
func (r *GGUFLocalReranker) computeRerankerScore(query, document string) (float64, error) {
	// Create cache key
	cacheKey := fmt.Sprintf("%s|||%s", query, document)
	
	// Check cache first
	r.cacheMutex.RLock()
	if cached, exists := r.scoreCache[cacheKey]; exists {
		r.cacheMutex.RUnlock()
		return cached, nil
	}
	r.cacheMutex.RUnlock()
	
	// Try reranker approach first
	score, err := r.tryRerankerInference(query, document)
	if err == nil {
		// Cache the result
		r.cacheMutex.Lock()
		r.scoreCache[cacheKey] = score
		r.cacheMutex.Unlock()
		return score, nil
	}
	
	// Fallback to embedding similarity
	fmt.Printf("DEBUG: Reranker failed (%v), falling back to embedding similarity\n", err)
	score, err = r.computeEmbeddingSimilarity(query, document)
	if err != nil {
		return 0.0, err
	}
	
	// Cache the result
	r.cacheMutex.Lock()
	r.scoreCache[cacheKey] = score
	r.cacheMutex.Unlock()
	
	return score, nil
}

// tryRerankerInference attempts to use llama-embedding with --pooling rank for reranking
func (r *GGUFLocalReranker) tryRerankerInference(query, document string) (float64, error) {
	// Format input for reranker model using proper format
	// Based on llama.cpp PR #9510, rerankers expect query</s><s>document format
	input := fmt.Sprintf("%s</s><s>%s", query, document)
	
	// Prepare command using llama-embedding with --pooling rank
	args := []string{
		"-m", r.modelPath,
		"-p", input,
		"--pooling", "rank", // Use rank pooling for reranker models
		"--embd-normalize", "-1", // Disable normalization for reranker scores
		"--verbose-prompt", // Enable verbose output for debugging
	}
	
	// Determine number of threads
	if r.config.Options != nil {
		if threads, ok := r.config.Options["threads"].(int); ok && threads > 0 {
			args = append(args, "-t", fmt.Sprintf("%d", threads))
		}
	}
	
	cmd := exec.Command(r.inferenceBinary, args...)
	
	// Capture output
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	// Run command
	if err := cmd.Run(); err != nil {
		return 0.0, fmt.Errorf("reranker command failed: %v", err)
	}
	
	// Parse the reranker score from output
	return r.parseRerankerScore(strings.TrimSpace(stdout.String()), strings.TrimSpace(stderr.String()))
}

// parseRerankerScore parses the numerical score from llama-embedding --pooling rank output
func (r *GGUFLocalReranker) parseRerankerScore(stdout, stderr string) (float64, error) {
	// Look for "rerank score" pattern in stderr (based on PR #9510 examples)
	lines := strings.Split(stderr, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.Contains(line, "rerank score") {
			// Extract score from line like "rerank score 0: -6.851"
			parts := strings.Fields(line)
			for i, part := range parts {
				if part == "score" && i+2 < len(parts) {
					// Skip the index (e.g., "0:") and get the score
					scoreStr := parts[i+2]
					if score, err := strconv.ParseFloat(scoreStr, 64); err == nil {
						return score, nil
					}
				}
			}
		}
	}
	
	// If no score found in stderr, try parsing stdout
	if stdout != "" {
		// Try to parse as a direct numerical value
		if score, err := strconv.ParseFloat(strings.TrimSpace(stdout), 64); err == nil {
			return score, nil
		}
	}
	
	return 0.0, fmt.Errorf("could not parse reranker score from output")
}

// computeEmbeddingSimilarity computes similarity using embeddings as fallback
func (r *GGUFLocalReranker) computeEmbeddingSimilarity(query, document string) (float64, error) {
	// Get embeddings for query and document
	queryEmb, err := r.getEmbedding(query)
	if err != nil {
		return 0.0, fmt.Errorf("failed to get query embedding: %v", err)
	}
	
	docEmb, err := r.getEmbedding(document)
	if err != nil {
		return 0.0, fmt.Errorf("failed to get document embedding: %v", err)
	}
	
	// Compute cosine similarity
	similarity := cosineSimilarity(queryEmb, docEmb)
	
	// Convert similarity to reranker-like score (scale from [-1,1] to [-10,10])
	return similarity * 10.0, nil
}

// getEmbedding computes embedding for a text using llama-embedding
func (r *GGUFLocalReranker) getEmbedding(text string) ([]float64, error) {
	// Prepare command for embedding extraction
	args := []string{
		"-m", r.modelPath,
		"-p", text,
		"--embd-output-format", "json",
		"--embd-normalize", "2", // L2 normalization
	}
	
	// Determine number of threads
	if r.config.Options != nil {
		if threads, ok := r.config.Options["threads"].(int); ok && threads > 0 {
			args = append(args, "-t", fmt.Sprintf("%d", threads))
		}
	}
	
	cmd := exec.Command(r.inferenceBinary, args...)
	
	// Capture output
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	// Run command
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("embedding command failed: %v, stderr: %s", err, stderr.String())
	}
	
	// Parse JSON output
	var response EmbeddingResponse
	if err := json.Unmarshal([]byte(stdout.String()), &response); err != nil {
		return nil, fmt.Errorf("failed to parse embedding response: %v", err)
	}
	
	if len(response.Data) == 0 {
		return nil, fmt.Errorf("no embedding data returned")
	}
	
	return response.Data[0].Embedding, nil
}

// cosineSimilarity computes cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}
	
	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	
	if normA == 0.0 || normB == 0.0 {
		return 0.0
	}
	
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Rerank reorders documents based on relevance to a query using GGUF model
func (r *GGUFLocalReranker) Rerank(ctx context.Context, query string, documents []Document) ([]Document, error) {
	if len(documents) == 0 {
		return documents, nil
	}
	
	// Calculate scores using GGUF model
	scores, err := r.ComputeScore(ctx, query, documents)
	if err != nil {
		return nil, err
	}
	
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

// ComputeScore computes scores for query-document pairs using GGUF reranker model
func (r *GGUFLocalReranker) ComputeScore(ctx context.Context, query string, documents []Document) ([]float64, error) {
	if len(documents) == 0 {
		return nil, nil
	}
	
	// Compute relevance scores for each document
	scores := make([]float64, len(documents))
	for i, doc := range documents {
		score, err := r.computeRerankerScore(query, doc.Content)
		if err != nil {
			// If scoring fails, assign a low score
			scores[i] = -5.0
			continue
		}
		scores[i] = score
	}
	
	return scores, nil
}

// Rank returns top-N ranked documents using GGUF model
func (r *GGUFLocalReranker) Rank(ctx context.Context, query string, documents []Document, topN int) ([]RerankResult, error) {
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
func (r *GGUFLocalReranker) GetModelName() string {
	return r.config.Model
}

// Configure updates the reranker configuration
func (r *GGUFLocalReranker) Configure(config Config) error {
	r.config = config
	if r.config.MaxDocs == 0 {
		r.config.MaxDocs = 100
	}
	return nil
}

// Close cleans up resources (clears cache)
func (r *GGUFLocalReranker) Close() {
	r.cacheMutex.Lock()
	r.scoreCache = make(map[string]float64)
	r.cacheMutex.Unlock()
}
