package onnx

import "testing"

func TestConfigNewRequiresLibraryPath(t *testing.T) {
	var cfg Config
	if err := cfg.New(); err == nil {
		t.Fatal("Config.New() error = nil, want an error when OnnxRuntimeLibPath is empty")
	}
}

func TestConfigCloseWithoutInit(t *testing.T) {
	var cfg Config
	cfg.Close() // must not panic
}
