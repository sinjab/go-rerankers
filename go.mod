module go-rerankers

go 1.21

// CGO is required for GGUF local inference via llama.cpp
// Build with: CGO_ENABLED=1 go build
//
// Dependencies will be added when HuggingFace and ONNX integrations are implemented
// require (
//   github.com/hupe1980/go-huggingface - for HF API integration
//   github.com/knights-analytics/hugot - for ONNX local inference
//   github.com/yalue/onnxruntime_go - for ONNX runtime bindings
// )
