package examples

import (
	ocr "github.com/getcharzp/go-ocr"
	"github.com/up-zero/gotool/imageutil"
	"log"
	"testing"
	"time"
)

func TestPaddleOcr(t *testing.T) {
	start := time.Now()
	config := ocr.Config{
		OnnxRuntimeLibPath: "../lib/onnxruntime.dll",
		DetModelPath:       "../paddle_weights/det.onnx",
		RecModelPath:       "../paddle_weights/rec.onnx",
		DictPath:           "../paddle_weights/dict.txt",
	}

	engine, err := ocr.NewPaddleOcrEngine(config)
	if err != nil {
		log.Fatalf("创建 OCR 引擎失败: %v\n", err)
	}

	defer engine.Destroy()

	imagePath := "./test.jpg"
	img, err := imageutil.Open(imagePath)
	if err != nil {
		log.Fatalf("加载图像失败: %v\n", err)
	}

	// 检测
	boxes, err := engine.RunDetect(img)
	if err != nil {
		log.Fatalf("运行检测失败: %v\n", err)
	}
	t.Logf("检测完成, 耗时：%v\n", time.Since(start))

	// 绘制检测区域
	detImage := ocr.DrawBoxes(img, boxes)
	imageutil.Save("det.jpg", detImage, 100)

	// 识别
	for _, box := range boxes {
		start2 := time.Now()
		result, err := engine.RunRecognize(img, box)
		if err != nil {
			log.Fatalf("运行识别失败: %v\n", err)
		}
		t.Logf("识别结果: %v, 耗时：%v\n", result, time.Since(start2))
	}

	t.Logf("测试完成，共识别 %d 个文本框, 耗时: %v\n", len(boxes), time.Since(start))
}
