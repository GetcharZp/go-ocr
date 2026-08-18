package paddle

import ort "github.com/getcharzp/onnxruntime_purego"

// Config PaddleOCR 引擎的开放配置
type Config struct {
	// 必填参数
	OnnxRuntimeLibPath string // onnxruntime.dll (或 .so, .dylib) 的路径
	DetModelPath       string // det.onnx (检测模型) 的路径
	RecModelPath       string // rec.onnx (识别模型) 的路径
	DictPath           string // dict.txt (字典) 的路径

	// 可选参数
	UseCuda             bool    // (可选) 是否启用 CUDA
	NumThreads          int     // (可选) ONNX 线程数, 默认由CPU核心数决定
	ThreadCount         int     // (可选) 识别 Session 并发数, 默认 1。>1 时创建多个 ONNX Session 实现并行识别
	DetMaxSideLen       int     // (可选) 检测模型预处理的最长边, 默认 960
	DetOutsideExpandPix int     // (可选) 检测框外扩像素, 默认 10
	RecHeight           int     // (可选) 识别模型预处理的高度, 默认 48
	RecModelNumClasses  int64   // (可选) 识别模型类别数, 默认 18385
	HeatmapThreshold    float32 // (可选) 检测热力图阈值, 默认 0.3
}

// Engine 是 PaddleOCR 引擎的主结构体
type Engine struct {
	detSession  *ort.Session   // 检测（单例，线程安全）
	recSessions []*ort.Session // 识别 session 池，支持并行

	dict                []string // 字典
	detMaxSideLen       int      // 检测模型最长边
	detOutsideExpandPix int      // 检测框外扩像素
	recHeight           int      // 识别模型高度
	recModelNumClasses  int64    // 识别模型类别数
	heatmapThreshold    float32  // 热力图阈值
}

// boundingBox 矩形
type boundingBox struct {
	MinX, MinY, MaxX, MaxY int
}
