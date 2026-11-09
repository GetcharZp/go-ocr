package ocr

import (
	"runtime"
	"testing"

	"github.com/getcharzp/go-ocr/internal/testutil"
)

func TestDefaultLibraryPath(t *testing.T) {
	var want string
	switch runtime.GOOS {
	case "windows":
		want = "./lib/onnxruntime.dll"
	case "darwin":
		want = "./lib/onnxruntime_" + runtime.GOARCH + ".dylib"
	case "linux":
		want = "./lib/onnxruntime_" + runtime.GOARCH + ".so"
	default:
		want = "./lib/onnxruntime_" + runtime.GOARCH + ".so"
	}

	if got := DefaultLibraryPath(); got != want {
		t.Fatalf("DefaultLibraryPath() = %q, want %q", got, want)
	}
}

func TestDefaultLibraryNameMatchesTestHelper(t *testing.T) {
	switch runtime.GOOS {
	case "windows", "darwin", "linux":
	default:
		t.Skipf("no platform-specific library naming on %s", runtime.GOOS)
	}

	if got, want := testutil.DefaultLibName(), libraryName(DefaultLibraryPath()); got != want {
		t.Fatalf("test helper library name %q does not match DefaultLibraryPath base name %q", got, want)
	}
}

func libraryName(path string) string {
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == '/' || path[i] == '\\' {
			return path[i+1:]
		}
	}
	return path
}
