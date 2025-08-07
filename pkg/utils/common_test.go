package utils

import (
	"testing"
	"go-rerankers/pkg/reranker"
)

func TestStringsToDocuments(t *testing.T) {
	docs := []string{"First document", "Second document", "Third document"}
	
	result := StringsToDocuments(docs)
	
	if len(result) != len(docs) {
		t.Errorf("Expected %d documents, got %d", len(docs), len(result))
	}
	
	for i, doc := range result {
		if doc.Content != docs[i] {
			t.Errorf("Expected content %s, got %s", docs[i], doc.Content)
		}
		
		if doc.ID == "" {
			t.Error("Expected non-empty ID")
		}
	}
}

func TestBenchmarkReranker(t *testing.T) {
	config := reranker.Config{
		Model:   "simple",
		MaxDocs: 10,
		Device:  "cpu",
	}
	
	r := reranker.NewSimpleReranker(config)
	
	documents := []reranker.Document{
		{ID: "1", Content: "Machine learning is powerful"},
		{ID: "2", Content: "Cooking is fun"},
		{ID: "3", Content: "AI and machine learning"},
	}
	
	query := "machine learning"
	
	result := BenchmarkReranker(r, query, documents, 1)
	
	if result == nil {
		t.Fatal("Expected benchmark result")
	}
	
	if result.ModelName != "simple" {
		t.Errorf("Expected model name 'simple', got %s", result.ModelName)
	}
	
	if result.NumDocs != len(documents) {
		t.Errorf("Expected %d docs, got %d", len(documents), result.NumDocs)
	}
	
	if result.Duration <= 0 {
		t.Error("Expected positive duration")
	}
	
	if result.DocsPerSec <= 0 {
		t.Error("Expected positive docs per second")
	}
}

func TestGetDevice(t *testing.T) {
	device := GetDevice()
	
	if device == "" {
		t.Error("Expected non-empty device string")
	}
	
	// Should return "cpu" for now as per implementation
	if device != "cpu" {
		t.Errorf("Expected 'cpu', got %s", device)
	}
}
