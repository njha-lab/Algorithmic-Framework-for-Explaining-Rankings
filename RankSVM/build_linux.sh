#!/bin/bash

set -e

echo "========================================"
echo "Building RankSVM for Linux"
echo "========================================"

echo "Cleaning previous build..."
make clean

echo "Compiling..."
make

echo "Copying executables..."

cp svm_rank_learn svm_rank/
cp svm_rank_classify svm_rank/

chmod +x svm_rank/svm_rank_learn
chmod +x svm_rank/svm_rank_classify

echo "========================================"
echo "Build completed successfully!"
echo "========================================"