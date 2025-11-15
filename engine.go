package ocr

import (
	"bufio"
	"fmt"
	"github.com/up-zero/gotool/imageutil"
	"image"
	"image/color"
	_ "image/jpeg" // 注册 jpeg 解码器
	"os"

	"golang.org/x/image/draw"
)

// RecResult OCR 识别结果结构体
type RecResult struct {
	Box   [4]int  // [x1, y1, x2, y2]
	Text  string  // 识别的文本
	Score float32 // 平均置信度
}

// Config OCR 引擎的开放配置
type Config struct {
	// 必填参数
	OnnxRuntimeLibPath string // onnxruntime.dll (或 .so, .dylib) 的路径
	DetModelPath       string // det.onnx (检测模型) 的路径
	RecModelPath       string // rec.onnx (识别模型) 的路径
	DictPath           string // dict.txt (字典) 的路径

	// 可选参数
	UseCuda             bool    // (可选) 是否启用 CUDA
	NumThreads          int     // (可选) ONNX 线程数, 默认由CPU核心数决定
	DetMaxSideLen       int     // (可选) 检测模型预处理的最长边, 默认 960
	DetOutsideExpandPix int     // (可选) 检测框外扩像素, 默认 10
	RecHeight           int     // (可选) 识别模型预处理的高度, 默认 48
	RecModelNumClasses  int64   // (可选) 识别模型类别数, 默认 18385
	HeatmapThreshold    float32 // (可选) 检测热力图阈值, 默认 0.3
}

// Engine 定义了 OCR 引擎必须实现的通用接口
type Engine interface {
	// RunDetect 图像文字区域检测
	RunDetect(img image.Image) ([][4]int, error)

	// RunRecognize 识别图像中指定区域的文字
	RunRecognize(img image.Image, box [4]int) (RecResult, error)

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

// boundingBox 定义一个简单的矩形
type boundingBox struct {
	MinX, MinY, MaxX, MaxY int
}

// postprocessHeatmap 将 ONNX 输出热力图转换为边界框
func postprocessHeatmap(heatmap []float32, h, w int64, origW, origH int, threshold float32, expandPix int) [][4]int {
	thresholdMap := make([][]bool, h)
	for y := 0; y < int(h); y++ {
		thresholdMap[y] = make([]bool, w)
		for x := 0; x < int(w); x++ {
			if heatmap[int64(y)*w+int64(x)] > threshold {
				thresholdMap[y][x] = true
			}
		}
	}

	visited := make([][]bool, h)
	for y := 0; y < int(h); y++ {
		visited[y] = make([]bool, w)
	}

	var boxes []boundingBox
	for y := 0; y < int(h); y++ {
		for x := 0; x < int(w); x++ {
			if thresholdMap[y][x] && !visited[y][x] {
				queue := [][2]int{{x, y}}
				visited[y][x] = true
				blobBox := boundingBox{MinX: x, MinY: y, MaxX: x, MaxY: y}

				for len(queue) > 0 {
					point := queue[0]
					queue = queue[1:]
					px, py := point[0], point[1]
					if px < blobBox.MinX {
						blobBox.MinX = px
					}
					if py < blobBox.MinY {
						blobBox.MinY = py
					}
					if px > blobBox.MaxX {
						blobBox.MaxX = px
					}
					if py > blobBox.MaxY {
						blobBox.MaxY = py
					}
					neighbors := [][2]int{{px + 1, py}, {px - 1, py}, {px, py + 1}, {px, py - 1}}
					for _, n := range neighbors {
						nx, ny := n[0], n[1]
						if nx >= 0 && nx < int(w) && ny >= 0 && ny < int(h) &&
							thresholdMap[ny][nx] && !visited[ny][nx] {
							visited[ny][nx] = true
							queue = append(queue, [2]int{nx, ny})
						}
					}
				}
				if (blobBox.MaxX-blobBox.MinX) > 5 && (blobBox.MaxY-blobBox.MinY) > 5 {
					blobBox.MinX = max(blobBox.MinX-expandPix, 0)
					blobBox.MinY = max(blobBox.MinY-expandPix, 0)
					blobBox.MaxX = min(blobBox.MaxX+expandPix, int(w)-1)
					blobBox.MaxY = min(blobBox.MaxY+expandPix, int(h)-1)
					boxes = append(boxes, blobBox)
				}
			}
		}
	}

	scaleX := float64(origW) / float64(w)
	scaleY := float64(origH) / float64(h)
	finalBoxes := make([][4]int, len(boxes))
	for i, box := range boxes {
		x1 := int(float64(box.MinX) * scaleX)
		y1 := int(float64(box.MinY) * scaleY)
		x2 := int(float64(box.MaxX) * scaleX)
		y2 := int(float64(box.MaxY) * scaleY)
		finalBoxes[i] = [4]int{x1, y1, x2, y2}
	}
	return finalBoxes
}

// loadDict 加载字典文件
func loadDict(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("无法打开字典文件 %s: %w", path, err)
	}
	defer file.Close()
	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("读取字典文件时出错: %w", err)
	}
	return lines, nil
}
