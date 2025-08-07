# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# Binary names
BINARY_NAME=go-rerankers
CLI_BINARY=reranker-cli
EXAMPLE_BINARY=basic-example
BGE_EXAMPLE_BINARY=bge-example

# Build targets
.PHONY: all build clean test deps example cli

all: deps test build

build:
	$(GOBUILD) -o $(BINARY_NAME) -v ./main.go

cli:
	$(GOBUILD) -o $(CLI_BINARY) -v ./cmd/cli/

example:
	$(GOBUILD) -o $(EXAMPLE_BINARY) -v ./examples/basic_usage.go

bge-example:
	$(GOBUILD) -o $(BGE_EXAMPLE_BINARY) -v ./examples/bge/main.go

test:
	$(GOTEST) -v ./...

clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME)
	rm -f $(CLI_BINARY)
	rm -f $(EXAMPLE_BINARY)
	rm -f $(BGE_EXAMPLE_BINARY)

deps:
	$(GOMOD) tidy
	$(GOMOD) download

run:
	$(GOCMD) run ./main.go

run-example:
	$(GOCMD) run ./examples/basic_usage.go

run-bge-example:
	$(GOCMD) run ./examples/bge/main.go

run-cli:
	$(GOCMD) run ./cmd/cli/ -query "machine learning" -input ./examples/sample_documents.json

fmt:
	$(GOCMD) fmt ./...

vet:
	$(GOCMD) vet ./...

lint: fmt vet

# Development helpers
dev: deps lint test

install:
	$(GOCMD) install ./cmd/cli/

# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Run deps, test, and build"
	@echo "  build     - Build the main binary"
	@echo "  cli       - Build the CLI binary"
	@echo "  example   - Build the example binary"
	@echo "  bge-example - Build the BGE reranker example binary"
	@echo "  test      - Run tests"
	@echo "  clean     - Clean build artifacts"
	@echo "  deps      - Download dependencies"
	@echo "  run       - Run the main application"
	@echo "  run-example - Run the basic usage example"
	@echo "  run-bge-example - Run the BGE reranker example"
	@echo "  run-cli   - Run the CLI with sample data (supports all BGE models)"
	@echo "  fmt       - Format code"
	@echo "  vet       - Run go vet"
	@echo "  lint      - Run fmt and vet"
	@echo "  dev       - Run deps, lint, and test"
	@echo "  install   - Install CLI globally"
	@echo "  help      - Show this help message"
