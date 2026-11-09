// Package testutil provides shared helpers for tests that need the ONNX Runtime
// shared library and the OCR model files.
//
// Asset lookup order:
//  1. explicit environment variables (see the Env* constants);
//  2. <repo>/.ci-assets, which is where scripts/download-assets.sh stores files;
//  3. the repository root itself (lib/, paddle_weights/, ddddocr_weights/, examples/).
//
// Tests that need an asset are skipped when it is missing, unless
// GO_OCR_REQUIRE_ASSETS=1 is set, in which case they fail. CI sets that variable
// so a broken download cannot silently turn the suite into a no-op.
package testutil

import (
	"fmt"
	"image"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/up-zero/gotool/imageutil"

	_ "image/jpeg"
	_ "image/png"
)

const (
	// EnvRoot overrides the repository root.
	EnvRoot = "GO_OCR_TEST_ROOT"
	// EnvAssetsDir overrides the directory that contains lib/, paddle_weights/ and ddddocr_weights/.
	EnvAssetsDir = "GO_OCR_ASSETS_DIR"
	// EnvLibPath points directly at the ONNX Runtime shared library.
	EnvLibPath = "GO_OCR_LIB_PATH"
	// EnvPaddleDir points at the directory containing det.onnx, rec.onnx and dict.txt.
	EnvPaddleDir = "GO_OCR_PADDLE_DIR"
	// EnvDdddocrDir points at the directory containing common.onnx, common_det.onnx and dict.txt.
	EnvDdddocrDir = "GO_OCR_DDDDOCR_DIR"
	// EnvExamplesDir points at the directory containing the example images.
	EnvExamplesDir = "GO_OCR_EXAMPLES_DIR"
	// EnvRequireAssets makes missing assets a test failure instead of a skip.
	EnvRequireAssets = "GO_OCR_REQUIRE_ASSETS"
)

// RepoRoot returns the root directory of the repository.
func RepoRoot() string {
	if dir := os.Getenv(EnvRoot); dir != "" {
		return dir
	}

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "."
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

// AssetsRoot returns the directory that holds lib/, paddle_weights/ and ddddocr_weights/.
func AssetsRoot() string {
	if dir := os.Getenv(EnvAssetsDir); dir != "" {
		return dir
	}

	ciAssets := filepath.Join(RepoRoot(), ".ci-assets")
	if info, err := os.Stat(ciAssets); err == nil && info.IsDir() {
		return ciAssets
	}
	return RepoRoot()
}

// DefaultLibName returns the ONNX Runtime library file name for the current platform.
func DefaultLibName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return fmt.Sprintf("onnxruntime_%s.dylib", runtime.GOARCH)
	default:
		return fmt.Sprintf("onnxruntime_%s.so", runtime.GOARCH)
	}
}

// LibPath returns the path to the ONNX Runtime shared library.
func LibPath() string {
	if path := os.Getenv(EnvLibPath); path != "" {
		return path
	}
	return filepath.Join(AssetsRoot(), "lib", DefaultLibName())
}

// PaddleDir returns the directory containing the PaddleOCR models and dictionary.
func PaddleDir() string {
	if dir := os.Getenv(EnvPaddleDir); dir != "" {
		return dir
	}
	return filepath.Join(AssetsRoot(), "paddle_weights")
}

// DdddocrDir returns the directory containing the ddddocr models and dictionary.
func DdddocrDir() string {
	if dir := os.Getenv(EnvDdddocrDir); dir != "" {
		return dir
	}
	return filepath.Join(AssetsRoot(), "ddddocr_weights")
}

// ExamplesDir returns the directory containing the example images.
func ExamplesDir() string {
	if dir := os.Getenv(EnvExamplesDir); dir != "" {
		return dir
	}
	return filepath.Join(RepoRoot(), "examples")
}

// ExamplePath returns the path to a file inside the examples directory.
func ExamplePath(name string) string {
	return filepath.Join(ExamplesDir(), name)
}

// RequireAsset skips (or fails, when GO_OCR_REQUIRE_ASSETS=1) if path is not a
// non-empty file. Short mode always skips asset-dependent tests.
func RequireAsset(t testing.TB, path string) {
	t.Helper()

	if testing.Short() {
		t.Skip("skipping asset-dependent test in short mode")
	}

	if info, err := os.Stat(path); err == nil && !info.IsDir() && info.Size() > 0 {
		return
	}

	const hint = "run `bash scripts/download-assets.sh .ci-assets` to download it"
	if os.Getenv(EnvRequireAssets) == "1" {
		t.Fatalf("required test asset is missing: %s (%s)", path, hint)
	}
	t.Skipf("required test asset is missing: %s (%s)", path, hint)
}

// OpenImage loads an example image, skipping the test when the file is absent.
func OpenImage(t testing.TB, path string) image.Image {
	t.Helper()
	RequireAsset(t, path)

	img, err := imageutil.Open(path)
	if err != nil {
		t.Fatalf("open image %s: %v", path, err)
	}
	return img
}
