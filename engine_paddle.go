package ocr

import (
	"fmt"
	"image"
	"math"
	"strings"
	"sync"

	"github.com/getcharzp/go-ocr/internal/onnx"
	"github.com/getcharzp/go-ocr/internal/util"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/imageutil"
	"golang.org/x/image/draw"
)

// PaddleOcrEngine 是 PaddleOCR 引擎的主结构体
type PaddleOcrEngine struct {
	detSession  *ort.Session   // 检测（单例，线程安全）
	recSessions []*ort.Session // 识别 session 池，支持并行

	dict                []string // 字典
	detMaxSideLen       int      // 检测模型最长边
	detOutsideExpandPix int      // 检测框外扩像素
	recHeight           int      // 识别模型高度
	recModelNumClasses  int64    // 识别模型类别数
	heatmapThreshold    float32  // 热力图阈值
}

// NewPaddleOcrEngine 用于初始化 ONNX Runtime、加载模型和字典。
func NewPaddleOcrEngine(cfg Config) (*PaddleOcrEngine, error) {
	oc := new(onnx.Config)
	_ = convertutil.CopyProperties(cfg, oc)

	if err := oc.New(); err != nil {
		return nil, err
	}

	// 检查和设置默认值
	if cfg.DetModelPath == "" || cfg.RecModelPath == "" || cfg.DictPath == "" {
		return nil, fmt.Errorf("模型路径和字典路径不能为空")
	}
	if cfg.DetMaxSideLen == 0 {
		cfg.DetMaxSideLen = 960
	}
	if cfg.RecHeight == 0 {
		cfg.RecHeight = 48
	}
	if cfg.RecModelNumClasses == 0 {
		cfg.RecModelNumClasses = 18385
	}
	if cfg.HeatmapThreshold == 0 {
		cfg.HeatmapThreshold = 0.3
	}
	if cfg.DetOutsideExpandPix == 0 {
		cfg.DetOutsideExpandPix = 10
	}
	if cfg.ThreadCount <= 0 {
		cfg.ThreadCount = 1
	}

	// 加载字典
	dict, err := util.LoadDict(cfg.DictPath)
	if err != nil {
		return nil, fmt.Errorf("加载字典失败: %w", err)
	}

	detSession, err := oc.OnnxEngine.NewSession(cfg.DetModelPath, oc.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建 det session 失败: %w", err)
	}

	// 创建识别 session 池
	recSessions := make([]*ort.Session, 0, cfg.ThreadCount)
	for i := 0; i < cfg.ThreadCount; i++ {
		session, err := oc.OnnxEngine.NewSession(cfg.RecModelPath, oc.SessionOptions)
		if err != nil {
			detSession.Destroy()
			for _, s := range recSessions {
				s.Destroy()
			}
			return nil, fmt.Errorf("创建 rec session[%d/%d] 失败: %w", i, cfg.ThreadCount, err)
		}
		recSessions = append(recSessions, session)
	}

	engine := &PaddleOcrEngine{
		detSession:          detSession,
		recSessions:         recSessions,
		dict:                dict,
		detMaxSideLen:       cfg.DetMaxSideLen,
		detOutsideExpandPix: cfg.DetOutsideExpandPix,
		recHeight:           cfg.RecHeight,
		recModelNumClasses:  cfg.RecModelNumClasses,
		heatmapThreshold:    cfg.HeatmapThreshold,
	}

	return engine, nil
}

// RunDetect 图像文字区域检测
func (e *PaddleOcrEngine) RunDetect(img image.Image) ([][4]int, error) {
	origBounds := img.Bounds()
	origWidth := origBounds.Dx()
	origHeight := origBounds.Dy()

	// 预处理
	detInputData, detInputShape := e.preprocessDetImage(img)
	detInputTensor, err := ort.NewTensor(detInputShape, detInputData)
	if err != nil {
		return nil, fmt.Errorf("创建 det input tensor 失败: %w", err)
	}
	defer detInputTensor.Destroy()
	detInputValues := map[string]*ort.Value{
		e.detSession.InputNames[0]: detInputTensor,
	}

	// 运行检测模型
	detOutputValues, err := e.detSession.Run(detInputValues)
	if err != nil {
		return nil, fmt.Errorf("运行 det session 失败: %w", err)
	}
	detOutputValue := detOutputValues[e.detSession.OutputNames[0]]
	defer detOutputValue.Destroy()

	detOutputData, err := ort.GetTensorData[float32](detOutputValue)
	if err != nil {
		return nil, fmt.Errorf("获取 det output data 失败: %w", err)
	}

	// 后处理热力图，获取边界框
	finalBoxes := postprocessHeatmap(
		detOutputData,
		detInputShape[2], // h
		detInputShape[3], // w
		origWidth,
		origHeight,
		e.heatmapThreshold,
		e.detOutsideExpandPix,
	)

	if len(finalBoxes) == 0 {
		return [][4]int{}, nil
	}

	return finalBoxes, nil
}

// runRecognize 识别图像中指定区域的文字（接受外部 session 实现无锁并发）
func (e *PaddleOcrEngine) runRecognize(session *ort.Session, img image.Image, box [4]int) (RecResult, error) {
	resultText := ""
	resultScore := float32(0.0)

	// 裁切
	crop, err := imageutil.Crop(img, image.Rectangle{
		Min: image.Point{X: box[0], Y: box[1]},
		Max: image.Point{X: box[2], Y: box[3]},
	})
	if err != nil {
		return RecResult{}, fmt.Errorf("裁切框失败: %w", err)
	}

	// 预处理
	recInputData, recInputShape := e.preprocessRecImage(crop)
	recInputTensor, err := ort.NewTensor(recInputShape, recInputData)
	if err != nil {
		return RecResult{}, fmt.Errorf("创建 rec input tensor 失败: %w", err)
	}
	defer recInputTensor.Destroy()
	recInputValues := map[string]*ort.Value{
		session.InputNames[0]: recInputTensor,
	}

	// 模型推理（session 专属，无需 Mutex）
	recOutputValues, err := session.Run(recInputValues)
	if err != nil {
		return RecResult{}, fmt.Errorf("运行 rec session 失败: %w", err)
	}
	recOutputValue := recOutputValues[session.OutputNames[0]]
	defer recOutputValue.Destroy()

	recOutputData, err := ort.GetTensorData[float32](recOutputValue)
	if err != nil {
		return RecResult{}, fmt.Errorf("获取 rec output data 错误: %w", err)
	}

	recOutputShape, err := recOutputValue.GetShape()
	if err != nil {
		return RecResult{}, fmt.Errorf("获取 rec output shape 错误: %w", err)
	}

	// 后处理 (CTC 解码)
	resultText, resultScore = e.postprocessRecOutput(recOutputData, recOutputShape)

	return RecResult{Box: box, Text: resultText, Score: resultScore}, nil
}

// RunOCR 对图像执行检测和识别
// 核心优化：Worker-Pool 模式 — 每个 rec session 独占一个 goroutine，按 stride 分摊 boxes
func (e *PaddleOcrEngine) RunOCR(img image.Image) ([]RecResult, error) {
	// 文字区域检测
	finalBoxes, err := e.RunDetect(img)
	if err != nil {
		return nil, err
	}

	numSessions := len(e.recSessions)
	if numSessions == 0 {
		return nil, fmt.Errorf("识别 session 未初始化")
	}

	results := make([]RecResult, len(finalBoxes))
	var errs []error
	var errMu sync.Mutex
	addErr := func(err error) {
		errMu.Lock()
		errs = append(errs, err)
		errMu.Unlock()
	}

	if len(finalBoxes) == 0 {
		return results, nil
	}

	// Worker-Pool: 每个 session 独占一个 goroutine，按 stride 分摊 boxes
	// 工作线程数不超过 boxes 数量，避免空转 goroutine
	workers := min(numSessions, len(finalBoxes))
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerIdx int) {
			defer wg.Done()
			session := e.recSessions[workerIdx]
			// 按 stride 分摊：worker 0 处理 box[0], box[workers], box[2*workers]...
			for j := workerIdx; j < len(finalBoxes); j += workers {
				result, err := e.runRecognize(session, img, finalBoxes[j])
				if err != nil {
					addErr(fmt.Errorf("识别框[%d] (box: %v) 错误: %w", j, finalBoxes[j], err))
					continue
				}
				results[j] = result
			}
		}(i)
	}

	wg.Wait()

	if len(errs) > 0 {
		return results, fmt.Errorf("部分识别失败 (%d/%d): %v", len(errs), len(finalBoxes), errs)
	}

	return results, nil
}

// Destroy 释放所有 ONNX 资源
func (e *PaddleOcrEngine) Destroy() {
	if e.detSession != nil {
		e.detSession.Destroy()
	}
	for _, session := range e.recSessions {
		if session != nil {
			session.Destroy()
		}
	}
}

// preprocessDetImage 对 PaddleOCR 检测模型进行预处理
func (e *PaddleOcrEngine) preprocessDetImage(img image.Image) ([]float32, []int64) {
	origBounds := img.Bounds()
	origWidth := origBounds.Dx()
	origHeight := origBounds.Dy()

	maxSideLen := e.detMaxSideLen
	ratio := 1.0
	if origWidth > origHeight {
		ratio = float64(maxSideLen) / float64(origWidth)
	} else {
		ratio = float64(maxSideLen) / float64(origHeight)
	}
	newWidth := int(math.Round(float64(origWidth) * ratio))
	newHeight := int(math.Round(float64(origHeight) * ratio))

	newWidth = (newWidth / 32) * 32
	newHeight = (newHeight / 32) * 32
	if newWidth == 0 {
		newWidth = 32
	}
	if newHeight == 0 {
		newHeight = 32
	}

	dstRect := image.Rect(0, 0, newWidth, newHeight)
	dstImg := image.NewRGBA(dstRect)
	draw.CatmullRom.Scale(dstImg, dstRect, img, origBounds, draw.Over, nil)

	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}

	shape := []int64{1, 3, int64(newHeight), int64(newWidth)}
	inputData := make([]float32, 1*3*newHeight*newWidth)

	for y := 0; y < newHeight; y++ {
		for x := 0; x < newWidth; x++ {
			r, g, b, _ := dstImg.At(x, y).RGBA()
			rNorm := float32(r>>8) / 255.0
			gNorm := float32(g>>8) / 255.0
			bNorm := float32(b>>8) / 255.0

			rFinal := (rNorm - mean[0]) / std[0]
			gFinal := (gNorm - mean[1]) / std[1]
			bFinal := (bNorm - mean[2]) / std[2]

			inputData[0*newHeight*newWidth+y*newWidth+x] = rFinal
			inputData[1*newHeight*newWidth+y*newWidth+x] = gFinal
			inputData[2*newHeight*newWidth+y*newWidth+x] = bFinal
		}
	}
	return inputData, shape
}

// preprocessRecImage 识别预处理
func (e *PaddleOcrEngine) preprocessRecImage(crop image.Image) ([]float32, []int64) {
	origBounds := crop.Bounds()
	origWidth := origBounds.Dx()
	origHeight := origBounds.Dy()
	recHeight := e.recHeight

	ratio := float64(recHeight) / float64(origHeight)
	newWidth := int(math.Round(float64(origWidth) * ratio))
	if newWidth == 0 {
		newWidth = 1
	}

	// 识别模型 (SVTR/v3/v4) 要求宽度是 8 的倍数
	if newWidth%8 != 0 {
		newWidth = ((newWidth / 8) + 1) * 8
	}
	if newWidth == 0 {
		newWidth = 8
	}

	dstRect := image.Rect(0, 0, newWidth, recHeight)
	dstImg := image.NewRGBA(dstRect)
	draw.CatmullRom.Scale(dstImg, dstRect, crop, origBounds, draw.Over, nil)

	shape := []int64{1, 3, int64(recHeight), int64(newWidth)}
	inputData := make([]float32, 1*3*recHeight*newWidth)
	mean := float32(0.5)
	std := float32(0.5)

	for y := 0; y < recHeight; y++ {
		for x := 0; x < newWidth; x++ {
			r, g, b, _ := dstImg.At(x, y).RGBA()
			rNorm := (float32(r>>8)/255.0 - mean) / std
			gNorm := (float32(g>>8)/255.0 - mean) / std
			bNorm := (float32(b>>8)/255.0 - mean) / std

			inputData[0*recHeight*newWidth+y*newWidth+x] = rNorm
			inputData[1*recHeight*newWidth+y*newWidth+x] = gNorm
			inputData[2*recHeight*newWidth+y*newWidth+x] = bNorm
		}
	}
	return inputData, shape
}

// postprocessRecOutput 后处理 (CTC 解码)
func (e *PaddleOcrEngine) postprocessRecOutput(output []float32, shape []int64) (string, float32) {
	wSeq := int(shape[1])
	numClasses := int(shape[2])

	var indices []int
	var scores []float32

	for i := 0; i < wSeq; i++ {
		stepOutput := output[i*numClasses : (i+1)*numClasses]
		maxScore := float32(-1e9)
		maxIdx := 0
		for j := 0; j < numClasses; j++ {
			if stepOutput[j] > maxScore {
				maxScore = stepOutput[j]
				maxIdx = j
			}
		}
		indices = append(indices, maxIdx)
		scores = append(scores, maxScore)
	}

	var decodedIndices []int
	lastIdx := -1
	totalScore := float32(0.0)
	count := 0
	for i, idx := range indices {
		// 索引 0 是 "blank" (dict.txt 中的 "__background__"), 跳过
		if idx > 0 && idx != lastIdx {
			decodedIndices = append(decodedIndices, idx)
			totalScore += scores[i]
			count++
		}
		lastIdx = idx
	}

	var strBuilder strings.Builder
	dict := e.dict
	for _, idx := range decodedIndices {
		// 字典索引 = 模型索引 - 1
		if idx > 0 && (idx-1) < len(dict) {
			strBuilder.WriteString(dict[idx-1])
		} else if idx != 0 {
			strBuilder.WriteString("?")
		}
	}

	avgScore := float32(0.0)
	if count > 0 {
		avgScore = totalScore / float32(count)
	}
	return strBuilder.String(), avgScore
}
