package examples

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/getcharzp/go-ocr/ddddocr"
	"github.com/getcharzp/go-ocr/internal/testutil"
)

const wantCaptcha = "f233g"

func newDdddocrEngine(t *testing.T, threadCount int) *ddddocr.Engine {
	t.Helper()

	lib := testutil.LibPath()
	model := filepath.Join(testutil.DdddocrDir(), "common.onnx")
	detModel := filepath.Join(testutil.DdddocrDir(), "common_det.onnx")
	dict := filepath.Join(testutil.DdddocrDir(), "dict.txt")
	for _, path := range []string{lib, model, detModel, dict} {
		testutil.RequireAsset(t, path)
	}

	engine, err := ddddocr.NewEngine(ddddocr.Config{
		OnnxRuntimeLibPath: lib,
		ModelPath:          model,
		DetModelPath:       detModel,
		DictPath:           dict,
		ThreadCount:        threadCount,
	})
	if err != nil {
		t.Fatalf("create ddddocr engine: %v", err)
	}
	t.Cleanup(engine.Destroy)
	return engine
}

func TestDdddOCRClassification(t *testing.T) {
	engine := newDdddocrEngine(t, 1)
	img := testutil.OpenImage(t, testutil.ExamplePath("captcha.png"))

	got, err := engine.Classification(img)
	if err != nil {
		t.Fatalf("Classification() error = %v", err)
	}
	if got != wantCaptcha {
		t.Fatalf("Classification() = %q, want %q", got, wantCaptcha)
	}
}

func TestDdddOCRDetect(t *testing.T) {
	engine := newDdddocrEngine(t, 1)
	img := testutil.OpenImage(t, testutil.ExamplePath("captcha_det.png"))
	bounds := img.Bounds()

	boxes, err := engine.Detect(img)
	if err != nil {
		t.Fatalf("Detect() error = %v", err)
	}
	if len(boxes) < 10 {
		t.Fatalf("Detect() returned %d boxes, want at least 10", len(boxes))
	}
	for i, box := range boxes {
		if box.Box[0] < 0 || box.Box[1] < 0 || box.Box[2] > bounds.Dx() || box.Box[3] > bounds.Dy() ||
			box.Box[0] >= box.Box[2] || box.Box[1] >= box.Box[3] {
			t.Errorf("box[%d] = %v is outside the image bounds %v", i, box.Box, bounds)
		}
		if box.Score <= 0 {
			t.Errorf("box[%d] has a non-positive score: %v", i, box.Score)
		}
	}
}

func detBoxSet(boxes []ddddocr.DetResult) map[[4]int]int {
	set := make(map[[4]int]int, len(boxes))
	for _, box := range boxes {
		set[box.Box]++
	}
	return set
}

// TestDdddOCRConcurrent calls Classification and Detect from many goroutines
// against one shared engine. This is the scenario that used to crash the
// Windows test run with an access violation (0xc0000005).
func TestDdddOCRConcurrent(t *testing.T) {
	engine := newDdddocrEngine(t, 2)
	captcha := testutil.OpenImage(t, testutil.ExamplePath("captcha.png"))
	detImg := testutil.OpenImage(t, testutil.ExamplePath("captcha_det.png"))

	baselineText, err := engine.Classification(captcha)
	if err != nil {
		t.Fatalf("baseline Classification() error = %v", err)
	}
	if baselineText != wantCaptcha {
		t.Fatalf("baseline Classification() = %q, want %q", baselineText, wantCaptcha)
	}
	baselineBoxes, err := engine.Detect(detImg)
	if err != nil {
		t.Fatalf("baseline Detect() error = %v", err)
	}
	if len(baselineBoxes) == 0 {
		t.Fatal("baseline Detect() returned no boxes")
	}
	wantBoxes := detBoxSet(baselineBoxes)

	const (
		goroutines = 32
		iterations = 20
	)
	var wg sync.WaitGroup
	errs := make(chan error, goroutines)

	for worker := range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range iterations {
				gotText, err := engine.Classification(captcha)
				if err != nil {
					errs <- fmt.Errorf("worker %d iteration %d: Classification() error: %w", worker, i, err)
					return
				}
				if gotText != baselineText {
					errs <- fmt.Errorf("worker %d iteration %d: Classification() = %q, want %q",
						worker, i, gotText, baselineText)
					return
				}

				if i%5 != 0 {
					continue
				}
				gotBoxes, err := engine.Detect(detImg)
				if err != nil {
					errs <- fmt.Errorf("worker %d iteration %d: Detect() error: %w", worker, i, err)
					return
				}
				if gotSet := detBoxSet(gotBoxes); !reflect.DeepEqual(gotSet, wantBoxes) {
					errs <- fmt.Errorf("worker %d iteration %d: Detect() boxes differ from baseline\n got: %v\nwant: %v",
						worker, i, gotSet, wantBoxes)
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
}

// TestDdddOCRClassificationCustomModel is opt-in because the model and charset
// come from dddd-trainer and are not part of the published asset repository.
func TestDdddOCRClassificationCustomModel(t *testing.T) {
	model := os.Getenv("GO_OCR_DDDDOCR_CUSTOM_MODEL")
	charset := os.Getenv("GO_OCR_DDDDOCR_CUSTOM_CHARSET")
	if model == "" || charset == "" {
		t.Skip("set GO_OCR_DDDDOCR_CUSTOM_MODEL and GO_OCR_DDDDOCR_CUSTOM_CHARSET to run")
	}

	lib := testutil.LibPath()
	for _, path := range []string{lib, model, charset} {
		testutil.RequireAsset(t, path)
	}

	engine, err := ddddocr.NewEngine(ddddocr.Config{
		OnnxRuntimeLibPath: lib,
		ModelPath:          model,
		DictPath:           charset,
		UseCustomModel:     true,
	})
	if err != nil {
		t.Fatalf("create custom-model ddddocr engine: %v", err)
	}
	defer engine.Destroy()

	img := testutil.OpenImage(t, testutil.ExamplePath("captcha.png"))
	got, err := engine.Classification(img)
	if err != nil {
		t.Fatalf("Classification() error = %v", err)
	}
	if got == "" {
		t.Fatal("Classification() returned an empty result")
	}
	t.Logf("custom model result: %q", got)
}
