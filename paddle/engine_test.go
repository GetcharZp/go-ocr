package paddle

import (
	"image"
	"math"
	"testing"

	ocr "github.com/getcharzp/go-ocr"
)

var _ ocr.Engine = (*Engine)(nil)

func TestNewEngineRequiresLibraryPath(t *testing.T) {
	if _, err := NewEngine(Config{}); err == nil {
		t.Fatal("NewEngine() error = nil, want an error when OnnxRuntimeLibPath is empty")
	}
}

func TestPostprocessRecOutput(t *testing.T) {
	engine := &Engine{dict: []string{"a", "b", "c"}}

	// 5 time steps x 4 classes. Index 0 is the CTC blank.
	steps := [][]float32{
		{-1, 0.9, -1, -1}, // class 1 -> "a"
		{-1, 0.9, -1, -1}, // repeated, skipped
		{-1, -1, -1, -1},  // blank, skipped
		{-1, -1, 0.8, -1}, // class 2 -> "b"
		{-1, -1, -1, 0.7}, // class 3 -> "c"
	}
	output := make([]float32, 0, len(steps)*4)
	for _, s := range steps {
		output = append(output, s...)
	}

	text, score := engine.postprocessRecOutput(output, []int64{1, int64(len(steps)), 4})
	if text != "abc" {
		t.Fatalf("postprocessRecOutput() text = %q, want %q", text, "abc")
	}
	wantScore := float32((0.9 + 0.8 + 0.7) / 3)
	if math.Abs(float64(score-wantScore)) > 1e-6 {
		t.Fatalf("postprocessRecOutput() score = %v, want %v", score, wantScore)
	}
}

func TestPostprocessHeatmap(t *testing.T) {
	const (
		h      = 16
		w      = 16
		origW  = 32
		origH  = 32
		expand = 2
	)
	heatmap := make([]float32, h*w)
	for y := 4; y <= 10; y++ {
		for x := 3; x <= 9; x++ {
			heatmap[y*w+x] = 1
		}
	}

	boxes := postprocessHeatmap(heatmap, h, w, origW, origH, 0.3, expand)
	if len(boxes) != 1 {
		t.Fatalf("postprocessHeatmap() returned %d boxes, want 1: %v", len(boxes), boxes)
	}

	// Blob [3,4]-[9,10] expanded by 2, then scaled by origW/w = 2.
	want := [4]int{2, 4, 22, 24}
	if boxes[0] != want {
		t.Fatalf("postprocessHeatmap() box = %v, want %v", boxes[0], want)
	}
}

func TestPostprocessHeatmapIgnoresSmallBlobs(t *testing.T) {
	const (
		h = 16
		w = 16
	)
	heatmap := make([]float32, h*w)
	heatmap[5*w+5] = 1 // single pixel, smaller than the 5px minimum

	if boxes := postprocessHeatmap(heatmap, h, w, 32, 32, 0.3, 0); len(boxes) != 0 {
		t.Fatalf("postprocessHeatmap() returned %d boxes, want 0: %v", len(boxes), boxes)
	}
}

func TestPreprocessDetImageShape(t *testing.T) {
	engine := &Engine{detMaxSideLen: 960}
	img := image.NewRGBA(image.Rect(0, 0, 100, 50))

	data, shape := engine.preprocessDetImage(img)
	if len(data) != 3*480*960 {
		t.Fatalf("preprocessDetImage() data length = %d, want %d", len(data), 3*480*960)
	}
	want := []int64{1, 3, 480, 960}
	for i := range want {
		if shape[i] != want[i] {
			t.Fatalf("preprocessDetImage() shape = %v, want %v", shape, want)
		}
	}
}

func TestPreprocessRecImageShape(t *testing.T) {
	engine := &Engine{recHeight: 48}
	img := image.NewRGBA(image.Rect(0, 0, 100, 50))

	data, shape := engine.preprocessRecImage(img)
	if len(data) != 3*48*96 {
		t.Fatalf("preprocessRecImage() data length = %d, want %d", len(data), 3*48*96)
	}
	want := []int64{1, 3, 48, 96}
	for i := range want {
		if shape[i] != want[i] {
			t.Fatalf("preprocessRecImage() shape = %v, want %v", shape, want)
		}
	}
}
