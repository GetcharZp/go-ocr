package ocr

import (
	"image"
	"image/color"
	"testing"
)

func TestDrawBoxes(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 20, 20))
	box := [4]int{2, 2, 10, 10}

	out := DrawBoxes(src, [][4]int{box})
	if out.Bounds() != src.Bounds() {
		t.Fatalf("DrawBoxes bounds = %v, want %v", out.Bounds(), src.Bounds())
	}

	// DrawRectOutline paints the rectangle edges; the top-left pixel must be red.
	r, g, b, a := out.At(box[0], box[1]).RGBA()
	wantRed := color.RGBA{R: 255, G: 0, B: 0, A: 255}
	wr, wg, wb, wa := wantRed.RGBA()
	if r != wr || g != wg || b != wb || a != wa {
		t.Fatalf("pixel at (%d,%d) = (%d,%d,%d,%d), want (%d,%d,%d,%d)",
			box[0], box[1], r, g, b, a, wr, wg, wb, wa)
	}

	// Pixels outside the box must stay untouched.
	if _, _, _, alpha := out.At(15, 15).RGBA(); alpha != 0 {
		t.Fatalf("pixel outside the box was modified: alpha = %d", alpha)
	}
}

func TestDrawBoxesEmpty(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 4, 4))
	out := DrawBoxes(src, nil)
	if out.Bounds() != src.Bounds() {
		t.Fatalf("DrawBoxes bounds = %v, want %v", out.Bounds(), src.Bounds())
	}
}
