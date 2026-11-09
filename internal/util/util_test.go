package util

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadDict(t *testing.T) {
	path := filepath.Join(t.TempDir(), "dict.txt")
	content := "a\nb\n\nc\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := LoadDict(path)
	if err != nil {
		t.Fatalf("LoadDict() error = %v", err)
	}
	want := []string{"a", "b", "", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("LoadDict() = %q, want %q", got, want)
	}
}

func TestLoadDictMissingFile(t *testing.T) {
	if _, err := LoadDict(filepath.Join(t.TempDir(), "missing.txt")); err == nil {
		t.Fatal("LoadDict() error = nil, want an error")
	}
}
