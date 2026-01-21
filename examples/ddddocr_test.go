package examples

import (
	"github.com/getcharzp/go-ocr/ddddocr"
	"github.com/up-zero/gotool/imageutil"
	"golang.org/x/image/draw"
	"image"
	"image/color"
	"log"
	"testing"
	"time"
)

func TestDdddOcr_Classification(t *testing.T) {
	start := time.Now()
	config := ddddocr.Config{
		OnnxRuntimeLibPath: "../lib/onnxruntime.dll",
		ModelPath:          "../ddddocr_weights/common.onnx",
		DictPath:           "../ddddocr_weights/dict.txt",
	}

	engine, err := ddddocr.NewEngine(config)
	if err != nil {
		log.Fatalf("创建 OCR 引擎失败: %v\n", err)
	}
	defer engine.Destroy()

	imagePath := "./captcha.png"
	img, err := imageutil.Open(imagePath)
	if err != nil {
		log.Fatalf("加载图像失败: %v\n", err)
	}

	// 识别
	res, err := engine.Classification(img)
	if err != nil {
		log.Fatalf("运行检测失败: %v\n", err)
	}
	t.Logf("识别完成, 耗时：%v, 识别内容：%v\n", time.Since(start), res)
}

func TestDdddOcr_Detect(t *testing.T) {
	start := time.Now()
	config := ddddocr.Config{
		OnnxRuntimeLibPath: "../lib/onnxruntime.dll",
		DetModelPath:       "../ddddocr_weights/common_det.onnx",
	}

	engine, err := ddddocr.NewEngine(config)
	if err != nil {
		log.Fatalf("创建 OCR 引擎失败: %v\n", err)
	}
	defer engine.Destroy()

	imagePath := "./captcha_det.png"
	img, err := imageutil.Open(imagePath)
	if err != nil {
		log.Fatalf("加载图像失败: %v\n", err)
	}

	boxes, err := engine.Detect(img)
	if err != nil {
		log.Fatalf("运行检测失败: %v\n", err)
	}
	t.Logf("识别完成, 耗时：%v, 识别内容：%v\n", time.Since(start), boxes)

	tagImg := image.NewRGBA(img.Bounds())
	draw.Draw(tagImg, img.Bounds(), img, image.Point{}, draw.Src)

	for _, box := range boxes {
		imageutil.DrawThickRectOutline(tagImg, image.Rectangle{Min: image.Point{X: box.Box[0], Y: box.Box[1]},
			Max: image.Point{X: box.Box[2], Y: box.Box[3]}}, color.Black, 2)
	}
	imageutil.Save("captcha_det_result.png", tagImg, 100)
}
