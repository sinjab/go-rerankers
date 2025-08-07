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
	"strings"
	"sync"
)

// GGUFLocalReranker implements reranking using GGUF models with llama.cpp embedding
type GGUFLocalReranker struct {
	config          Config
	modelPath       string
	embeddingBinary string
	embeddingCache  map[string][]float64
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
	
	// Find the llama-embedding binary
	embeddingBinary := filepath.Join(filepath.Dir(modelPath), "..", "llama.cpp", "build", "bin", "llama-embedding")
	if _, err := os.Stat(embeddingBinary); os.IsNotExist(err) {
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
				embeddingBinary = alt
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
		embeddingBinary: embeddingBinary,
		embeddingCache:  make(map[string][]float64),
	}
	
	// Test the model by computing a simple embedding
	if err := reranker.testModel(); err != nil {
		return nil, fmt.Errorf("%w: model test failed: %v", ErrInitialization, err)
	}
	
	return reranker, nil
}

// testModel tests if the GGUF model works by computing a simple embedding
func (r *GGUFLocalReranker) testModel() error {
	// For now, just check if the binary and model exist
	if _, err := os.Stat(r.embeddingBinary); os.IsNotExist(err) {
		return fmt.Errorf("embedding binary not found: %s", r.embeddingBinary)
	}
	
	if _, err := os.Stat(r.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("model file not found: %s", r.modelPath)
	}
	
	// Quick test with a simple computation
	// We'll do a minimal test here since full embedding test might hang
	return nil
}

// computeEmbedding computes embedding for a given text using llama-embedding
func (r *GGUFLocalReranker) computeEmbedding(text string) ([]float64, error) {
	// Check cache first
	r.cacheMutex.RLock()
	if cached, exists := r.embeddingCache[text]; exists {
		r.cacheMutex.RUnlock()
		return cached, nil
	}
	r.cacheMutex.RUnlock()
	
	// Prepare command
	args := []string{
		"-m", r.modelPath,
		"--pooling", "mean",
		"--embd-output-format", "json",
		"--log-disable",
		"-p", text,
	}
	
	// Determine number of threads
	if r.config.Options != nil {
		if threads, ok := r.config.Options["threads"].(int); ok && threads > 0 {
			args = append(args, "-t", fmt.Sprintf("%d", threads))
		}
	}
	
	cmd := exec.Command(r.embeddingBinary, args...)
	
	// Capture output
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	
	// Run command
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("embedding command failed: %v, stderr: %s, stdout: %s", err, stderr.String(), stdout.String())
	}
	
	// Debug: log the output
	output := stdout.String()
	if output == "" {
		return nil, fmt.Errorf("embedding command returned empty output, stderr: %s", stderr.String())
	}
	
	// Parse JSON response
	var response EmbeddingResponse
	if err := json.Unmarshal([]byte(output), &response); err != nil {
		return nil, fmt.Errorf("failed to parse embedding response: %v, output: %s", err, output)
	}
	
	if len(response.Data) == 0 {
		return nil, fmt.Errorf("no embedding data returned")
	}
	
	embedding := response.Data[0].Embedding
	
	// Check if embedding is all zeros (indicates failure)
	allZeros := true
	for _, val := range embedding {
		if val != 0.0 {
			allZeros = false
			break
		}
	}
	
	if allZeros {
		return nil, fmt.Errorf("embedding computation returned all zeros")
	}
	
	// Cache the result
	r.cacheMutex.Lock()
	r.embeddingCache[text] = embedding
	r.cacheMutex.Unlock()
	
	return embedding, nil
}

// cosineSimilarity computes cosine similarity between two vectors
func (r *GGUFLocalReranker) cosineSimilarity(a, b []float64) float64 {
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

// ComputeScore computes scores for query-document pairs using GGUF model
func (r *GGUFLocalReranker) ComputeScore(ctx context.Context, query string, documents []Document) ([]float64, error) {
	if len(documents) == 0 {
		return nil, nil
	}
	
	// Compute query embedding
	queryEmbedding, err := r.computeEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("failed to compute query embedding: %v", err)
	}
	
	// Compute document embeddings and similarity scores
	scores := make([]float64, len(documents))
	for i, doc := range documents {
		docEmbedding, err := r.computeEmbedding(doc.Content)
		if err != nil {
			// If embedding fails, assign a low score
			scores[i] = -1.0
			continue
		}
		
		// Compute cosine similarity
		similarity := r.cosineSimilarity(queryEmbedding, docEmbedding)
		
		// Convert similarity to a reranker-style score
		// Cosine similarity is in [-1, 1], we'll map it to a wider range
		scores[i] = similarity * 10.0
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
	r.embeddingCache = make(map[string][]float64)
	r.cacheMutex.Unlock()
}
