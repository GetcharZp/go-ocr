#!/usr/bin/env bash
#
# Download the ONNX Runtime shared library and the OCR models used by the tests.
#
# Usage:
#   bash scripts/download-assets.sh [destination]   # default: .ci-assets
#
# Files are fetched from the HuggingFace repository first and fall back to the
# ModelScope mirror, so the script works both inside and outside mainland China.
# Existing files with a plausible size are skipped, which makes the script cheap
# to re-run locally and user-friendly with actions/cache in CI.

set -euo pipefail

dest="${1:-.ci-assets}"

hf_base="${GO_OCR_HF_BASE:-https://huggingface.co/getcharzp/go-ocr/resolve/main}"
ms_base="${GO_OCR_MS_BASE:-https://www.modelscope.cn/models/getcharzp/go-ocr/resolve/master}"

goos="$(go env GOOS)"
goarch="$(go env GOARCH)"

case "${goos}-${goarch}" in
windows-amd64)
	lib="onnxruntime.dll"
	;;
linux-amd64 | linux-arm64)
	lib="onnxruntime_${goarch}.so"
	;;
darwin-amd64 | darwin-arm64)
	lib="onnxruntime_${goarch}.dylib"
	;;
*)
	echo "unsupported platform: ${goos}-${goarch}" >&2
	exit 1
	;;
esac

files=(
	"lib/${lib}"
	"paddle_weights/det.onnx"
	"paddle_weights/rec.onnx"
	"paddle_weights/dict.txt"
	"ddddocr_weights/common.onnx"
	"ddddocr_weights/common_det.onnx"
	"ddddocr_weights/dict.txt"
)

# min_size returns a conservative lower bound for a file. It rejects HTML error
# pages and LFS pointers that some mirrors return with a 200 status.
min_size() {
	case "$1" in
	*dict.txt) echo 10000 ;;
	*.onnx) echo 5000000 ;;
	*.dll | *.so | *.dylib) echo 5000000 ;;
	*) echo 1 ;;
	esac
}

size_of() {
	wc -c <"$1" | tr -d ' '
}

download() {
	local rel="$1" out="$2" url size min
	min="$(min_size "$rel")"

	for url in "${hf_base}/${rel}" "${ms_base}/${rel}"; do
		echo "Downloading ${url}"
		if curl -fsSL --retry 3 --retry-delay 2 --retry-all-errors \
			--connect-timeout 30 --max-time 1800 \
			-o "${out}.part" "$url"; then
			size="$(size_of "${out}.part")"
			if [ "$size" -ge "$min" ]; then
				mv "${out}.part" "$out"
				echo "  ok (${size} bytes)"
				return 0
			fi
			echo "  file is too small (${size} bytes), trying the next mirror" >&2
		else
			echo "  download failed, trying the next mirror" >&2
		fi
		rm -f "${out}.part"
	done

	echo "ERROR: failed to download ${rel}" >&2
	return 1
}

mkdir -p "${dest}/lib" "${dest}/paddle_weights" "${dest}/ddddocr_weights"

for rel in "${files[@]}"; do
	out="${dest}/${rel}"
	if [ -f "$out" ] && [ "$(size_of "$out")" -ge "$(min_size "$rel")" ]; then
		echo "Skip ${rel} (already present)"
		continue
	fi
	download "$rel" "$out"
done

echo "Assets are ready in ${dest}"
