package util

import (
	"bufio"
	"fmt"
	"os"
)

// LoadDict 加载字典文件
func LoadDict(path string) ([]string, error) {
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
