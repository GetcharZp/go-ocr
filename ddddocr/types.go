package ddddocr

import ort "github.com/getcharzp/onnxruntime_purego"

// DetResult 检测结果结构体
type DetResult struct {
	Box   [4]int // [x1, y1, x2, y2]
	Score float32
}

// Config ddddocr 配置信息
type Config struct {
	ModelPath          string
	DetModelPath       string
	DictPath           string
	OnnxRuntimeLibPath string
	UseCustomModel     bool // true = 使用自定义模型 (dddd-trainer)
}

// Engine ddddocr 引擎
type Engine struct {
	ocrSession     *ort.Session
	detSession     *ort.Session
	dict           []string
	useCustomModel bool
}
