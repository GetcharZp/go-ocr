package examples

import (
	"fmt"
	"image"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	ocr "github.com/getcharzp/go-ocr"
	"github.com/getcharzp/go-ocr/internal/testutil"
	"github.com/getcharzp/go-ocr/paddle"
	"github.com/up-zero/gotool/imageutil"
)

func newPaddleEngine(t *testing.T, threadCount int) *paddle.Engine {
	t.Helper()

	lib := testutil.LibPath()
	detModel := filepath.Join(testutil.PaddleDir(), "det.onnx")
	recModel := filepath.Join(testutil.PaddleDir(), "rec.onnx")
	dict := filepath.Join(testutil.PaddleDir(), "dict.txt")
	for _, path := range []string{lib, detModel, recModel, dict} {
		testutil.RequireAsset(t, path)
	}

	engine, err := paddle.NewEngine(paddle.Config{
		OnnxRuntimeLibPath: lib,
		DetModelPath:       detModel,
		RecModelPath:       recModel,
		DictPath:           dict,
		ThreadCount:        threadCount,
		NumThreads:         2,
	})
	if err != nil {
		t.Fatalf("create paddle engine: %v", err)
	}
	t.Cleanup(engine.Destroy)
	return engine
}

func paddleResultKeys(results []ocr.RecResult) map[string]int {
	keys := make(map[string]int, len(results))
	for _, result := range results {
		keys[fmt.Sprintf("%v %s", result.Box, result.Text)]++
	}
	return keys
}

func TestPaddleRunDetectAndOCR(t *testing.T) {
	engine := newPaddleEngine(t, 1)
	img := testutil.OpenImage(t, testutil.ExamplePath("test.jpg"))
	bounds := img.Bounds()

	boxes, err := engine.RunDetect(img)
	if err != nil {
		t.Fatalf("RunDetect() error = %v", err)
	}
	if len(boxes) < 5 {
		t.Fatalf("RunDetect() returned %d boxes, want at least 5", len(boxes))
	}
	for i, box := range boxes {
		if box[0] < 0 || box[1] < 0 || box[2] > bounds.Dx() || box[3] > bounds.Dy() ||
			box[0] >= box[2] || box[1] >= box[3] {
			t.Errorf("box[%d] = %v is outside the image bounds %v", i, box, bounds)
		}
	}

	results, err := engine.RunOCR(img)
	if err != nil {
		t.Fatalf("RunOCR() error = %v", err)
	}
	if len(results) < 10 {
		t.Fatalf("RunOCR() returned %d results, want at least 10", len(results))
	}

	var sb strings.Builder
	recognized := 0
	for _, result := range results {
		sb.WriteString(result.Text)
		if result.Text != "" && result.Score > 0 {
			recognized++
		}
	}
	if recognized < 10 {
		t.Errorf("only %d of %d results have text, want at least 10", recognized, len(results))
	}

	allText := sb.String()
	for _, want := range []string{"护发素", "YM-X-3011", "产品信息"} {
		if !strings.Contains(allText, want) {
			t.Errorf("RunOCR() output does not contain %q: %q", want, allText)
		}
	}
}

// TestPaddleConcurrentRunOCR exercises the two supported concurrency modes:
// a single shared session (ThreadCount=1) and a session pool (ThreadCount>1).
// The same engine is called from several goroutines at once, which is how the
// library is used in production.
func TestPaddleConcurrentRunOCR(t *testing.T) {
	img := testutil.OpenImage(t, testutil.ExamplePath("test.jpg"))
	// The top half is enough to exercise detection and several recognition boxes
	// while keeping the stress test fast.
	crop, err := imageutil.Crop(img, image.Rect(0, 0, 640, 300))
	if err != nil {
		t.Fatalf("crop image: %v", err)
	}

	tests := []struct {
		name        string
		threadCount int
	}{
		{name: "shared-session", threadCount: 1},
		{name: "session-pool", threadCount: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := newPaddleEngine(t, tt.threadCount)

			baseline, err := engine.RunOCR(crop)
			if err != nil {
				t.Fatalf("baseline RunOCR() error = %v", err)
			}
			want := paddleResultKeys(baseline)
			if len(want) == 0 {
				t.Fatal("baseline RunOCR() returned no results")
			}

			const (
				goroutines = 4
				iterations = 2
			)
			var wg sync.WaitGroup
			errs := make(chan error, goroutines)

			for worker := range goroutines {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := range iterations {
						got, err := engine.RunOCR(crop)
						if err != nil {
							errs <- fmt.Errorf("worker %d iteration %d: RunOCR() error: %w", worker, i, err)
							return
						}
						if gotKeys := paddleResultKeys(got); !reflect.DeepEqual(gotKeys, want) {
							errs <- fmt.Errorf("worker %d iteration %d: results differ from baseline\n got: %v\nwant: %v",
								worker, i, gotKeys, want)
							return
						}
					}
				}()
			}

			wg.Wait()
			close(errs)
			for err := range errs {
				t.Error(err)
			}
		})
	}
}
