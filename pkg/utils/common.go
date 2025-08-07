package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"time"

	"go-rerankers/pkg/reranker"
)

// TestData represents the structure of test JSON files
type TestData struct {
	Query       string   `json:"query"`
	Documents   []string `json:"documents"`
	Instruction string   `json:"instruction,omitempty"`
}

// LoadTestData loads test data from a JSON file
func LoadTestData(filePath string) (*TestData, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read test file: %w", err)
	}

	var testData TestData
	if err := json.Unmarshal(data, &testData); err != nil {
		return nil, fmt.Errorf("failed to parse test file: %w", err)
	}

	return &testData, nil
}

// StringsToDocuments converts string slice to Document slice
func StringsToDocuments(docs []string) []reranker.Document {
	documents := make([]reranker.Document, len(docs))
	for i, doc := range docs {
		documents[i] = reranker.Document{
			ID:      fmt.Sprintf("doc_%d", i+1),
			Content: doc,
		}
	}
	return documents
}
// GetDevice detects the best available device for inference
func GetDevice() string {
	// TODO: Add actual device detection logic
	// For now, return "cpu" as default
	switch runtime.GOOS {
	case "darwin":
		// On macOS, check for Metal Performance Shaders availability
		return "cpu" // Default to CPU for now
	case "linux", "windows":
		// Check for CUDA availability
		return "cpu" // Default to CPU for now
	default:
		return "cpu"
	}
}

// BenchmarkResult represents the result of a benchmark run
type BenchmarkResult struct {
	ModelName   string        `json:"model_name"`
	Duration    time.Duration `json:"duration"`
	DocsPerSec  float64       `json:"docs_per_sec"`
	AvgScore    float64       `json:"avg_score"`
	NumDocs     int           `json:"num_docs"`
	Error       string        `json:"error,omitempty"`
}

// BenchmarkReranker runs a performance benchmark on a reranker
func BenchmarkReranker(r reranker.Reranker, query string, documents []reranker.Document, iterations int) *BenchmarkResult {
	if iterations <= 0 {
		iterations = 1
	}

	result := &BenchmarkResult{
		ModelName: r.GetModelName(),
		NumDocs:   len(documents),
	}

	start := time.Now()
	var totalScore float64
	var successfulRuns int

	for i := 0; i < iterations; i++ {
		ranked, err := r.Rank(nil, query, documents, len(documents))
		if err != nil {
			result.Error = err.Error()
			break
		}

		// Calculate average score for this run
		var runScore float64
		for _, res := range ranked {
			runScore += res.Score
		}
		if len(ranked) > 0 {
			runScore /= float64(len(ranked))
			totalScore += runScore
			successfulRuns++
		}
	}

	duration := time.Since(start)
	result.Duration = duration

	if successfulRuns > 0 {
		result.AvgScore = totalScore / float64(successfulRuns)
		docsProcessed := float64(result.NumDocs * successfulRuns)
		result.DocsPerSec = docsProcessed / duration.Seconds()
	}

	return result
}

// PrintResults prints reranking results in a formatted way
func PrintResults(modelName string, results []reranker.RerankResult, topK int) {
	fmt.Printf("\n=== %s Results ===\n", modelName)
	
	limit := len(results)
	if topK > 0 && topK < limit {
		limit = topK
	}

	for i := 0; i < limit; i++ {
		result := results[i]
		fmt.Printf("%d. [%.4f] %s\n", i+1, result.Score, result.Document.Content)
	}
}

// PrintBenchmark prints benchmark results in a formatted way
func PrintBenchmark(result *BenchmarkResult) {
	fmt.Printf("\n=== Benchmark: %s ===\n", result.ModelName)
	if result.Error != "" {
		fmt.Printf("Error: %s\n", result.Error)
		return
	}
	
	fmt.Printf("Duration: %v\n", result.Duration)
	fmt.Printf("Documents processed: %d\n", result.NumDocs)
	fmt.Printf("Docs/second: %.2f\n", result.DocsPerSec)
	fmt.Printf("Average score: %.4f\n", result.AvgScore)
}
