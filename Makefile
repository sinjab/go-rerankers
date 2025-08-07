# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod

# Binary names
BINARY_NAME=go-rerankers

# Build targets
.PHONY: all build clean test deps

all: deps test build

build:
	$(GOBUILD) -o $(BINARY_NAME) -v ./main.go



test:
	$(GOTEST) -v ./...

clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME)

deps:
	$(GOMOD) tidy
	$(GOMOD) download

run:
	$(GOCMD) run ./main.go



fmt:
	$(GOCMD) fmt ./...

vet:
	$(GOCMD) vet ./...

lint: fmt vet

# Development helpers
dev: deps lint test



# Help target
help:
	@echo "Available targets:"
	@echo "  all       - Run deps, test, and build"
	@echo "  build     - Build the main binary"
	@echo "  test      - Run tests"
	@echo "  clean     - Clean build artifacts"
	@echo "  deps      - Download dependencies"
	@echo "  run       - Run the main application"
	@echo "  fmt       - Format code"
	@echo "  vet       - Run go vet"
	@echo "  lint      - Run fmt and vet"
	@echo "  dev       - Run deps, lint, and test"
	@echo "  help      - Show this help message"
