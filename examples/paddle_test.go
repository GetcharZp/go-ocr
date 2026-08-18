package examples

import (
	"log"
	"testing"
	"time"

	ocr "github.com/getcharzp/go-ocr"
	"github.com/getcharzp/go-ocr/paddle"
	"github.com/up-zero/gotool/imageutil"
)

func TestPaddleOcr(t *testing.T) {
	start := time.Now()
	config := paddle.Config{
		OnnxRuntimeLibPath: "../lib/onnxruntime.dll",
		DetModelPath:       "../paddle_weights/det.onnx",
		RecModelPath:       "../paddle_weights/rec.onnx",
		DictPath:           "../paddle_weights/dict.txt",
	}

	engine, err := paddle.NewEngine(config)
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

	// 识别（使用 RunOCR 完整流程）
	results, err := engine.RunOCR(img)
	if err != nil {
		log.Fatalf("运行识别失败: %v\n", err)
	}
	for _, result := range results {
		t.Logf("识别结果: %v\n", result)
	}

	t.Logf("测试完成，共识别 %d 个文本框, 耗时: %v\n", len(results), time.Since(start))
}
