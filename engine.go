package ocr

import (
	"image"
	"image/color"
	_ "image/jpeg" // 注册 jpeg 解码器

	"github.com/up-zero/gotool/imageutil"
	"golang.org/x/image/draw"
)

// RecResult OCR 识别结果结构体
type RecResult struct {
	Box   [4]int  // [x1, y1, x2, y2]
	Text  string  // 识别的文本
	Score float32 // 平均置信度
}

// Engine 定义了 OCR 引擎必须实现的通用接口
type Engine interface {
	// RunDetect 图像文字区域检测
	RunDetect(img image.Image) ([][4]int, error)

	// RunOCR 对图像执行检测和识别
	RunOCR(img image.Image) ([]RecResult, error)

	// Destroy 释放所有引擎相关的资源
	Destroy()
}

// DrawBoxes 在图像上绘制检测区域
func DrawBoxes(img image.Image, boxes [][4]int) image.Image {
	bounds := img.Bounds()
	drawImg := image.NewRGBA(bounds)
	draw.Draw(drawImg, bounds, img, image.Point{}, draw.Src)
	red := color.RGBA{R: 255, G: 0, B: 0, A: 255}

	for _, box := range boxes {
		imageutil.DrawRectOutline(drawImg, image.Rectangle{
			Min: image.Point{X: box[0], Y: box[1]},
			Max: image.Point{X: box[2], Y: box[3]},
		}, red)
	}
	return drawImg
}
