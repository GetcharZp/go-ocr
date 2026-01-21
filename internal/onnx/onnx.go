package onnx

import (
	"fmt"
	ort "github.com/getcharzp/onnxruntime_purego"
	"sync"
)

type Config struct {
	SessionOptions *ort.SessionOptions
	OnnxEngine     *ort.Engine

	// 必填参数
	OnnxRuntimeLibPath string // onnxruntime.dll (或 .so, .dylib) 的路径
	// 可选参数
	UseCuda    bool // (可选) 是否启用 CUDA
	NumThreads int  // (可选) ONNX 线程数, 默认由CPU核心数决定

	// EnableCpuMemArena 控制 ONNX 的内存池策略
	// false (默认): 禁用内存池，推理速度稍慢，但 Destroy 后立即归还内存给 OS ，解决内存滞留问题
	// true: 启用内存池，推理速度最快，但 Destroy 后内存会被缓存以供复用
	EnableCpuMemArena bool
}

var (
	initErr    error
	once       sync.Once
	onnxEngine *ort.Engine
)

// New 初始化 ONNX 环境
func (cfg *Config) New() error {
	// 初始化 ONNX Runtime
	if cfg.OnnxRuntimeLibPath == "" {
		return fmt.Errorf("OnnxRuntimeLibPath 不能为空")
	}
	once.Do(func() {
		onnxEngine, initErr = ort.NewEngine(cfg.OnnxRuntimeLibPath)
	})
	if initErr != nil {
		return fmt.Errorf("初始化 ONNX Engine 失败: %w", initErr)
	}

	// 创建会话选项 (设置线程)
	options, err := onnxEngine.NewSessionOptions()
	if err != nil {
		return err
	}
	if cfg.NumThreads > 0 {
		if err := options.SetIntraOpNumThreads(int32(cfg.NumThreads)); err != nil {
			return err
		}
	}

	// 设置内存策略
	if err := options.SetCpuMemArena(cfg.EnableCpuMemArena); err != nil {
		return fmt.Errorf("设置 CPU 内存池失败: %w", err)
	}

	// 启用CUDA
	if cfg.UseCuda {
		if err := options.EnableCUDA(); err != nil {
			return fmt.Errorf("启用 CUDA 失败: %w", err)
		}
	}
	cfg.SessionOptions = options
	cfg.OnnxEngine = onnxEngine

	return nil
}
