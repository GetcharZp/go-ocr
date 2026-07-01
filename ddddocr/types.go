package ddddocr

import (
	"sync/atomic"

	ort "github.com/getcharzp/onnxruntime_purego"
)

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
	ThreadCount        int  // (可选) OCR Session 并发数, 默认 1。>1 时创建多个 ONNX Session 实现并行识别
}

// Engine ddddocr 引擎
type Engine struct {
	ocrSessions    []*ort.Session // OCR session 池，支持无锁并发
	detSession     *ort.Session   // 检测 session（单例）
	sessionIdx     atomic.Uint64  // session 轮询计数器
	dict           []string
	useCustomModel bool
}
