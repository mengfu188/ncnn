#!/usr/bin/env bash

onnx_file="/home/cmf/w_public/Face-Detector-1MB-with-landmark/faceDetector.onnx"
onnx_slim_file="/home/cmf/w_public/Face-Detector-1MB-with-landmark/faceDetector-slim.onnx"
out_param="/home/cmf/w_public/Face-Detector-1MB-with-landmark/faceDetector.param"
out_bin="/home/cmf/w_public/Face-Detector-1MB-with-landmark/faceDetector.bin"
slim_param="/home/cmf/w_public/Face-Detector-1MB-with-landmark/faceDetector-slim.param"
slim_bin="/home/cmf/w_public/Face-Detector-1MB-with-landmark/faceDetector-slim.bin"

# Error
# [ValidationError: Your model ir_version is higher than the checker's](https://github.com/daquexian/onnx-simplifier/issues/2)
# onnx version=1.6.0
python3 -m onnxsim ${onnx_file} ${onnx_slim_file}

./onnx/onnx2ncnn ${onnx_slim_file} ${out_param} ${out_bin}

./ncnnoptimize ${out_param} ${out_bin} ${slim_param} ${slim_bin} 1