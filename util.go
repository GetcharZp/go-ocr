package ocr

import (
	"fmt"
	"runtime"
)

// DefaultLibraryPath 根据运行时环境判断加载哪个库文件
func DefaultLibraryPath() string {
	baseDir := "./lib/"
	libName := "onnxruntime"

	// windows onnxruntime.dll
	if runtime.GOOS == "windows" {
		return baseDir + libName + ".dll"
	}

	// linux darwin ext
	var ext string
	switch runtime.GOOS {
	case "darwin":
		ext = "dylib"
	case "linux":
		ext = "so"
	default:
		return baseDir + libName + "_amd64.so" // 默认返回 linux amd64
	}

	// 拼接完整路径: ./lib/onnxruntime + _ + amd64/arm64 + . + so/dylib
	return fmt.Sprintf("%s%s_%s.%s", baseDir, libName, runtime.GOARCH, ext)
}
