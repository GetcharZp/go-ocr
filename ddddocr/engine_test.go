package ddddocr

import (
	"reflect"
	"testing"

	ort "github.com/getcharzp/onnxruntime_purego"
)

func TestNewEngineRequiresLibraryPath(t *testing.T) {
	if _, err := NewEngine(Config{}); err == nil {
		t.Fatal("NewEngine() error = nil, want an error when OnnxRuntimeLibPath is empty")
	}
}

func TestAcquireSessionEmptyPool(t *testing.T) {
	engine := &Engine{}
	if session := engine.acquireSession(); session != nil {
		t.Fatalf("acquireSession() = %v, want nil for an empty pool", session)
	}
}

func TestAcquireSessionRoundRobin(t *testing.T) {
	first := &ort.Session{Inputs: []ort.TensorInfo{{Name: "first"}}}
	second := &ort.Session{Inputs: []ort.TensorInfo{{Name: "second"}}}
	engine := &Engine{ocrSessions: []*ort.Session{first, second}}

	got := []*ort.Session{
		engine.acquireSession(),
		engine.acquireSession(),
		engine.acquireSession(),
		engine.acquireSession(),
	}
	want := []*ort.Session{first, second, first, second}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("acquireSession() sequence = %v, want %v", got, want)
	}
}

func TestCalculateIOU(t *testing.T) {
	tests := []struct {
		name string
		a, b DetResult
		want float32
	}{
		{
			name: "identical",
			a:    DetResult{Box: [4]int{0, 0, 10, 10}},
			b:    DetResult{Box: [4]int{0, 0, 10, 10}},
			want: 1,
		},
		{
			name: "disjoint",
			a:    DetResult{Box: [4]int{0, 0, 10, 10}},
			b:    DetResult{Box: [4]int{20, 20, 30, 30}},
			want: 0,
		},
		{
			name: "half overlap",
			a:    DetResult{Box: [4]int{0, 0, 10, 10}},
			b:    DetResult{Box: [4]int{5, 0, 15, 10}},
			want: 1.0 / 3.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := calculateIOU(tt.a, tt.b)
			if diff := got - tt.want; diff > 1e-6 || diff < -1e-6 {
				t.Fatalf("calculateIOU() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNMS(t *testing.T) {
	boxes := []DetResult{
		{Box: [4]int{0, 0, 10, 10}, Score: 0.9},
		{Box: [4]int{1, 1, 11, 11}, Score: 0.8}, // overlaps the first box, suppressed
		{Box: [4]int{20, 20, 30, 30}, Score: 0.7},
	}

	got := nms(boxes)
	if len(got) != 2 {
		t.Fatalf("nms() returned %d boxes, want 2: %v", len(got), got)
	}
	if got[0].Score != 0.9 || got[1].Score != 0.7 {
		t.Fatalf("nms() scores = %v, %v, want 0.9, 0.7", got[0].Score, got[1].Score)
	}
}

func TestNMSEmpty(t *testing.T) {
	if got := nms(nil); got != nil {
		t.Fatalf("nms(nil) = %v, want nil", got)
	}
}

func TestPostprocessOCR(t *testing.T) {
	engine := &Engine{dict: []string{"a", "b", "c"}}
	output := make([]float32, 2*numClasses)
	output[0*numClasses+1] = 1 // first step: index 1 -> dict[1] = "b"
	output[1*numClasses+2] = 1 // second step: index 2 -> dict[2] = "c"

	if got := engine.postprocessOCR(output, 2); got != "bc" {
		t.Fatalf("postprocessOCR() = %q, want %q", got, "bc")
	}
}

func TestPostprocessOCRRepeatedClass(t *testing.T) {
	engine := &Engine{dict: []string{"a", "b", "c"}}
	output := make([]float32, 3*numClasses)
	output[0*numClasses+1] = 1
	output[1*numClasses+1] = 1
	output[2*numClasses+2] = 1

	if got := engine.postprocessOCR(output, 3); got != "bc" {
		t.Fatalf("postprocessOCR() = %q, want %q (repeated class must be collapsed)", got, "bc")
	}
}

func TestPostprocessOCRCustom(t *testing.T) {
	engine := &Engine{dict: []string{"zero", "a", "b", "c"}}

	// Repeated index 1 and blank 0 are skipped.
	if got := engine.postprocessOCRCustom([]int64{1, 1, 0, 2, 3}); got != "abc" {
		t.Fatalf("postprocessOCRCustom() = %q, want %q", got, "abc")
	}
}

func TestPostprocessOCRCustomOutOfRange(t *testing.T) {
	engine := &Engine{dict: []string{"a"}}
	if got := engine.postprocessOCRCustom([]int64{5}); got != "" {
		t.Fatalf("postprocessOCRCustom() = %q, want empty string for out-of-range index", got)
	}
}
