package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"go-rerankers/pkg/reranker"
	"go-rerankers/pkg/utils"
)

func main() {
	// Define CLI flags
	var (
		testFile   = flag.String("test-file", "", "Path to JSON test file")
		testAll    = flag.Bool("test-all", false, "Test all JSON files in test_data directory")
		query      = flag.String("query", "", "Query string (if not using test file)")
		documents  = flag.String("documents", "", "Comma-separated document strings (if not using test file)")
		modelName  = flag.String("reranker", "", "Specific reranker to use (default: all)")
		topK       = flag.Int("top-k", 3, "Number of top results to return")
		benchmark  = flag.Bool("benchmark", false, "Run performance benchmark instead of normal ranking")
		listModels = flag.Bool("list-models", false, "List all available models")
	)
	flag.Parse()

	// List models if requested
	if *listModels {
		printAvailableModels()
		return
	}

	// Test all JSON files if requested
	if *testAll {
		testAllJSONFiles(*modelName, *topK, *benchmark)
		return
	}

	// Get query and documents
	var queryStr string
	var docs []string

	if *testFile != "" {
		testData, err := utils.LoadTestData(*testFile)
		if err != nil {
			log.Fatalf("Error loading test file: %v", err)
		}
		queryStr = testData.Query
		docs = testData.Documents
	} else if *query != "" && *documents != "" {
		queryStr = *query
		docs = strings.Split(*documents, ",")
		// Trim whitespace from each document
		for i := range docs {
			docs[i] = strings.TrimSpace(docs[i])
		}
	} else {
		fmt.Println("Error: Either --test-file, --test-all, or both --query and --documents must be provided")
		fmt.Println("\nUsage examples:")
		fmt.Println("  go run main.go --test-file test_data/test_ml.json --top-k 3")
		fmt.Println("  go run main.go --test-all --reranker mxbai-v2 --top-k 3")
		fmt.Println("  go run main.go --query \"What is AI?\" --documents \"AI is...,Cooking...\" --reranker mxbai-v2")
		fmt.Println("  go run main.go --benchmark --reranker all")
		fmt.Println("  go run main.go --list-models")
		os.Exit(1)
	}

	fmt.Printf("Query: %s\n", queryStr)
	fmt.Printf("Number of documents: %d\n", len(docs))

	// Convert strings to documents
	documentList := utils.StringsToDocuments(docs)

	// Get device info
	device := utils.GetDevice()
	fmt.Printf("Using device: %s\n", device)

	if *benchmark {
		runBenchmark(queryStr, documentList, *modelName)
	} else {
		runReranking(queryStr, documentList, *modelName, *topK)
	}
}

func printAvailableModels() {
	fmt.Println("Available reranker models:")
	fmt.Println("=========================")
	
	models := reranker.GetSupportedModels()
	for _, model := range models {
		fmt.Printf("\nName: %s\n", model.Name)
		fmt.Printf("  Display Name: %s\n", model.DisplayName)
		fmt.Printf("  Provider: %s\n", model.Provider)
		fmt.Printf("  Model ID: %s\n", model.ModelID)
		fmt.Printf("  Type: %s\n", model.Type)
		if len(model.Strengths) > 0 {
			fmt.Printf("  Strengths: %s\n", strings.Join(model.Strengths, ", "))
		}
	}
}
func runReranking(query string, documents []reranker.Document, modelName string, topK int) {
	if modelName == "" || modelName == "all" {
		// Test all models
		testAllModels(query, documents, topK)
	} else {
		// Test specific model
		testSingleModel(query, documents, modelName, topK)
	}
}

func runBenchmark(query string, documents []reranker.Document, modelName string) {
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("RUNNING BENCHMARKS")
	fmt.Println(strings.Repeat("=", 50))

	var results []*utils.BenchmarkResult

	if modelName == "" || modelName == "all" {
		// Benchmark all models
		models := reranker.GetSupportedModels()
		for _, model := range models {
			result := benchmarkModel(query, documents, model.ModelID)
			if result != nil {
				results = append(results, result)
			}
		}
	} else {
		// Benchmark specific model
		result := benchmarkModel(query, documents, modelName)
		if result != nil {
			results = append(results, result)
		}
	}

	// Print benchmark summary
	if len(results) > 0 {
		fmt.Println("\n" + strings.Repeat("=", 50))
		fmt.Println("BENCHMARK SUMMARY")
		fmt.Println(strings.Repeat("=", 50))

		// Sort by duration (fastest first)
		for i := 0; i < len(results); i++ {
			for j := i + 1; j < len(results); j++ {
				if results[j].Duration < results[i].Duration {
					results[i], results[j] = results[j], results[i]
				}
			}
		}

		fmt.Println("\nReranker Performance (fastest to slowest):")
		for i, result := range results {
			if result.Error == "" {
				fmt.Printf("  %d. %s: %.4f seconds (%.2f docs/sec)\n", 
					i+1, result.ModelName, result.Duration.Seconds(), result.DocsPerSec)
			} else {
				fmt.Printf("  %d. %s: ERROR - %s\n", i+1, result.ModelName, result.Error)
			}
		}
	}
}

func testAllModels(query string, documents []reranker.Document, topK int) {
	models := reranker.GetSupportedModels()
	successCount := 0
	
	for _, model := range models {
		fmt.Printf("\n%s\n", strings.Repeat("=", 60))
		fmt.Printf("Testing: %s (%s)\n", model.DisplayName, model.Name)
		fmt.Printf("%s\n", strings.Repeat("=", 60))

		if testSingleModel(query, documents, model.ModelID, topK) {
			successCount++
		}
	}

	fmt.Printf("\n%s\n", strings.Repeat("=", 60))
	fmt.Printf("SUMMARY: %d/%d models tested successfully\n", successCount, len(models))
	fmt.Printf("%s\n", strings.Repeat("=", 60))
}

func testSingleModel(query string, documents []reranker.Document, modelName string, topK int) bool {
	config := reranker.Config{
		Model:     modelName,
		MaxDocs:   100,
		Threshold: -10.0, // Show all documents including low-scoring ones
		Device:    utils.GetDevice(),
	}

	r, err := reranker.NewReranker(config)
	if err != nil {
		fmt.Printf("Error initializing reranker: %v\n", err)
		return false
	}

	ctx := context.Background()
	start := time.Now()
	
	results, err := r.Rank(ctx, query, documents, topK)
	if err != nil {
		fmt.Printf("Error ranking documents: %v\n", err)
		return false
	}

	duration := time.Since(start)
	fmt.Printf("Ranking completed in %v\n", duration)

	utils.PrintResults(r.GetModelName(), results, topK)
	return true
}

func benchmarkModel(query string, documents []reranker.Document, modelName string) *utils.BenchmarkResult {
	config := reranker.Config{
		Model:     modelName,
		MaxDocs:   100,
		Threshold: -10.0,
		Device:    utils.GetDevice(),
	}

	r, err := reranker.NewReranker(config)
	if err != nil {
		return &utils.BenchmarkResult{
			ModelName: modelName,
			Error:     err.Error(),
		}
	}

	fmt.Printf("Benchmarking: %s...\n", r.GetModelName())
	
	// Run benchmark with 3 iterations for more accurate timing
	result := utils.BenchmarkReranker(r, query, documents, 3)
	
	utils.PrintBenchmark(result)
	return result
}

func testAllJSONFiles(modelName string, topK int, benchmark bool) {
	testDataDir := "test_data"
	
	// Get all JSON files in test_data directory
	files, err := filepath.Glob(filepath.Join(testDataDir, "*.json"))
	if err != nil {
		log.Fatalf("Error reading test_data directory: %v", err)
	}
	
	if len(files) == 0 {
		fmt.Println("No JSON files found in test_data directory")
		return
	}
	
	fmt.Printf("Found %d JSON test files in %s directory\n", len(files), testDataDir)
	fmt.Printf("%s\n", strings.Repeat("=", 80))
	
	successCount := 0
	totalFiles := len(files)
	
	for i, file := range files {
		fmt.Printf("\n[%d/%d] Testing file: %s\n", i+1, totalFiles, filepath.Base(file))
		fmt.Printf("%s\n", strings.Repeat("-", 60))
		
		// Load test data
		testData, err := utils.LoadTestData(file)
		if err != nil {
			fmt.Printf("❌ Error loading test file %s: %v\n", filepath.Base(file), err)
			continue
		}
		
		fmt.Printf("Query: %s\n", testData.Query)
		fmt.Printf("Documents: %d\n", len(testData.Documents))
		
		// Convert strings to documents
		documentList := utils.StringsToDocuments(testData.Documents)
		
		if benchmark {
			// Run benchmark for this file
			if modelName == "" || modelName == "all" {
				fmt.Println("\nRunning benchmarks for all models...")
				runBenchmark(testData.Query, documentList, modelName)
			} else {
				fmt.Printf("\nRunning benchmark for model: %s...\n", modelName)
				runBenchmark(testData.Query, documentList, modelName)
			}
			successCount++
		} else {
			// Run normal reranking for this file
			if modelName == "" || modelName == "all" {
				fmt.Println("\nTesting with all models...")
				testAllModels(testData.Query, documentList, topK)
			} else {
				fmt.Printf("\nTesting with model: %s...\n", modelName)
				if testSingleModel(testData.Query, documentList, modelName, topK) {
					successCount++
				}
			}
		}
		
		fmt.Printf("✅ Completed testing file: %s\n", filepath.Base(file))
	}
	
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	if benchmark {
		fmt.Printf("SUMMARY: Completed benchmarking %d test files\n", totalFiles)
	} else {
		fmt.Printf("SUMMARY: %d/%d test files processed successfully\n", successCount, totalFiles)
	}
	fmt.Printf("%s\n", strings.Repeat("=", 80))
}
