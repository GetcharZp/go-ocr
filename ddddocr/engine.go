package ddddocr

import (
	"fmt"
	"image"
	"math"
	"sort"
	"strings"

	"github.com/getcharzp/go-ocr/internal/onnx"
	"github.com/getcharzp/go-ocr/internal/util"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/imageutil"
)

const (
	numClasses     = 8210
	detInputSize   = 416
	iouThreshold   = 0.45
	scoreThreshold = 0.1
)

// NewEngine 初始化引擎
func NewEngine(cfg Config) (*Engine, error) {
	oc := new(onnx.Config)
	_ = convertutil.CopyProperties(cfg, oc)

	if err := oc.New(); err != nil {
		return nil, err
	}

	engine := &Engine{
		useCustomModel: cfg.UseCustomModel,
	}

	if cfg.ModelPath != "" {
		session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
		if err != nil {
			return nil, fmt.Errorf("创建 OCR 会话失败: %w", err)
		}
		engine.ocrSession = session

		dict, err := util.LoadDict(cfg.DictPath)
		if err != nil {
			return nil, fmt.Errorf("加载字符集失败: %w", err)
		}
		engine.dict = dict
	}

	if cfg.DetModelPath != "" {
		session, err := oc.OnnxEngine.NewSession(cfg.DetModelPath, oc.SessionOptions)
		if err != nil {
			return nil, fmt.Errorf("创建检测会话失败: %w", err)
		}
		engine.detSession = session
	}

	return engine, nil
}

// Classification 验证码识别
func (e *Engine) Classification(img image.Image) (string, error) {
	if e.ocrSession == nil {
		return "", fmt.Errorf("OCR 引擎未初始化")
	}

	inputData, inputShape, err := e.preprocessOCR(img)
	if err != nil {
		return "", err
	}

	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return "", err
	}
	defer inputTensor.Destroy()

	inputValues := map[string]*ort.Value{
		"input1": inputTensor,
	}

	outputValues, err := e.ocrSession.Run(inputValues)
	if err != nil {
		return "", fmt.Errorf("OCR 推理失败: %w", err)
	}

	if e.useCustomModel {
		// 自定义模型：输出节点 "output"，类型 int64，直接 CTC 解码
		outputValue := outputValues["output"]
		defer outputValue.Destroy()

		outputData, err := ort.GetTensorData[int64](outputValue)
		if err != nil {
			return "", fmt.Errorf("获取 OCR 输出数据失败: %w", err)
		}

		return e.postprocessOCRCustom(outputData), nil
	}

	// 官方模型：输出节点 "387"，类型 float32，argmax + CTC 解码
	outputValue := outputValues["387"]
	defer outputValue.Destroy()

	outputData, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return "", fmt.Errorf("获取 OCR 输出数据失败: %w", err)
	}

	seqLen := int(math.Ceil(float64(inputShape[3]) / 8.0))
	return e.postprocessOCR(outputData, seqLen), nil
}

// Detect 目标检测
func (e *Engine) Detect(img image.Image) ([]DetResult, error) {
	if e.detSession == nil {
		return nil, fmt.Errorf("检测引擎未初始化")
	}

	inputData, ratio := e.preprocessDet(img)

	inputShape := []int64{1, 3, detInputSize, detInputSize}
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	inputValues := map[string]*ort.Value{
		"images": inputTensor,
	}

	outputValues, err := e.detSession.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("检测推理失败: %w", err)
	}
	outputValue := outputValues["output"]
	defer outputValue.Destroy()

	outputData, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, err
	}

	return e.postprocessDet(outputData, ratio, img.Bounds().Dx(), img.Bounds().Dy()), nil
}

func (e *Engine) preprocessOCR(img image.Image) ([]float32, []int64, error) {
	targetH := 64
	dstImg := imageutil.Resize(img, 0, targetH)
	targetW := dstImg.Bounds().Dx()

	grayImg := imageutil.Grayscale(dstImg)
	inputData := make([]float32, 1*1*targetH*targetW)

	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			pix := grayImg.Pix[y*grayImg.Stride+x]
			normalized := float32(pix) / 255.0

			if e.useCustomModel {
				// 自定义模型：mean=0.456, std=0.224
				inputData[y*targetW+x] = (normalized - 0.456) / 0.224
			} else {
				// 官方模型：mean=0.5, std=0.5
				inputData[y*targetW+x] = (normalized - 0.5) / 0.5
			}
		}
	}

	shape := []int64{1, 1, int64(targetH), int64(targetW)}
	return inputData, shape, nil
}

func (e *Engine) postprocessOCR(output []float32, seqLen int) string {
	var sb strings.Builder
	lastIdx := -1

	for i := 0; i < seqLen; i++ {
		start := i * numClasses
		end := start + numClasses
		if end > len(output) {
			break
		}

		stepData := output[start:end]
		maxIdx := 0
		maxVal := float32(-1e9)
		for idx, val := range stepData {
			if val > maxVal {
				maxVal = val
				maxIdx = idx
			}
		}

		if maxIdx != 0 && maxIdx != lastIdx {
			if maxIdx >= 0 && maxIdx < len(e.dict) {
				sb.WriteString(e.dict[maxIdx])
			}
		}
		lastIdx = maxIdx
	}
	return sb.String()
}

// postprocessOCRCustom 自定义模型后处理（直接 CTC 解码，无需 argmax）
func (e *Engine) postprocessOCRCustom(output []int64) string {
	var sb strings.Builder
	lastIdx := int64(-1)

	for _, idx := range output {
		// CTC 解码：跳过空白符（index 0）和重复字符
		if idx != 0 && idx != lastIdx {
			if int(idx) < len(e.dict) {
				sb.WriteString(e.dict[idx])
			}
		}
		lastIdx = idx
	}

	return sb.String()
}

func (e *Engine) preprocessDet(img image.Image) (data []float32, ratio float64) {
	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()

	ratio = min(float64(detInputSize)/float64(srcH), float64(detInputSize)/float64(srcW))

	newW := int(float64(srcW) * ratio)
	newH := int(float64(srcH) * ratio)

	resized := imageutil.Resize(img, newW, newH)

	data = make([]float32, 1*3*detInputSize*detInputSize)
	area := detInputSize * detInputSize

	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			data[0*area+y*detInputSize+x] = float32(r >> 8)
			data[1*area+y*detInputSize+x] = float32(g >> 8)
			data[2*area+y*detInputSize+x] = float32(b >> 8)
		}
	}

	return data, ratio
}

func (e *Engine) postprocessDet(output []float32, ratio float64, imgW, imgH int) []DetResult {
	// [1, 3549, 6]
	strides := []int{8, 16, 32}
	var grids []float32
	var expandedStrides []float32

	for _, stride := range strides {
		hsize := detInputSize / stride
		wsize := detInputSize / stride
		for y := 0; y < hsize; y++ {
			for x := 0; x < wsize; x++ {
				grids = append(grids, float32(x), float32(y))
				expandedStrides = append(expandedStrides, float32(stride))
			}
		}
	}

	numAnchors := len(output) / 6
	var candidates []DetResult

	for i := 0; i < numAnchors; i++ {
		offset := i * 6
		objConf := output[offset+4]
		clsConf := output[offset+5]
		score := objConf * clsConf

		if score < scoreThreshold {
			continue
		}

		regX := (output[offset+0] + grids[i*2+0]) * expandedStrides[i]
		regY := (output[offset+1] + grids[i*2+1]) * expandedStrides[i]
		regW := float32(math.Exp(float64(output[offset+2]))) * expandedStrides[i]
		regH := float32(math.Exp(float64(output[offset+3]))) * expandedStrides[i]

		x1 := max((regX-regW/2)/float32(ratio), 0)
		y1 := max((regY-regH/2)/float32(ratio), 0)
		x2 := min((regX+regW/2)/float32(ratio), float32(imgW))
		y2 := min((regY+regH/2)/float32(ratio), float32(imgH))

		candidates = append(candidates, DetResult{
			Box: [4]int{
				int(x1),
				int(y1),
				int(x2),
				int(y2),
			},
			Score: score,
		})
	}

	return nms(candidates)
}

func nms(boxes []DetResult) []DetResult {
	if len(boxes) == 0 {
		return nil
	}
	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Score > boxes[j].Score
	})

	var result []DetResult
	for len(boxes) > 0 {
		current := boxes[0]
		result = append(result, current)
		boxes = boxes[1:]

		var remaining []DetResult
		for _, b := range boxes {
			if calculateIOU(current, b) < iouThreshold {
				remaining = append(remaining, b)
			}
		}
		boxes = remaining
	}
	return result
}

func calculateIOU(a, b DetResult) float32 {
	ix1 := max(float64(a.Box[0]), float64(b.Box[0]))
	iy1 := max(float64(a.Box[1]), float64(b.Box[1]))
	ix2 := min(float64(a.Box[2]), float64(b.Box[2]))
	iy2 := min(float64(a.Box[3]), float64(b.Box[3]))

	iw := max(0, ix2-ix1)
	ih := max(0, iy2-iy1)
	inter := iw * ih

	areaA := float64((a.Box[2] - a.Box[0]) * (a.Box[3] - a.Box[1]))
	areaB := float64((b.Box[2] - b.Box[0]) * (b.Box[3] - b.Box[1]))

	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return float32(inter / union)
}

func (e *Engine) Destroy() {
	if e.ocrSession != nil {
		e.ocrSession.Destroy()
	}
	if e.detSession != nil {
		e.detSession.Destroy()
	}
}
