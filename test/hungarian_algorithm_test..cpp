#include <iostream>
#include <cstring>
#include <vector>
#include "../src/Cuda/hungarian_algorithm.h"
#include <cuda_runtime.h>

#define N 5  // 行列の最大サイズ

int main() {
    HungarianAlgorithm hungarianAlgorithmInstance;

    // もとのコスト行列 (3x4 の行列を例に)
    int originalRows = 3, originalCols = 4;
    float h_matrix[] = {9, 2, 7, 8,
                      6, 4, 3, 7,
                      5, 8, 1, 8};

    // 正方行列のサイズを決定
    int newSize = std::max(originalRows, originalCols);

    // 新しい行列を作成してダミーを追加
    float *h_newMatrix = new float[newSize * newSize];
    memset(h_newMatrix, 0, newSize * newSize * sizeof(float));

    // 元の行列を新しい行列にコピー
    for (int i = 0; i < originalRows; ++i) {
        for (int j = 0; j < originalCols; ++j) {
            h_newMatrix[i * newSize + j] = h_matrix[i * originalCols + j];
        }
    }

    // ダミー要素を追加
    hungarianAlgorithmInstance.addDummyElements(h_newMatrix, originalRows, originalCols, newSize);

    // デバイスメモリのポインタ
    float *d_matrix, *d_minRow, *d_minCol;
    cudaMalloc((void**)&d_matrix, newSize * newSize * sizeof(int));
    cudaMalloc((void**)&d_minRow, newSize * sizeof(int));
    cudaMalloc((void**)&d_minCol, newSize * sizeof(int));

    // データをホストからデバイスにコピー
    cudaMemcpy(d_matrix, h_newMatrix, newSize * newSize * sizeof(int), cudaMemcpyHostToDevice);

    // カーネルを実行するホスト関数を呼び出し
    hungarianAlgorithmInstance.runHungarianAlgorithm(d_matrix, d_minRow, d_minCol, newSize);

    // 結果をホストにコピー
    cudaMemcpy(h_newMatrix, d_matrix, newSize * newSize * sizeof(int), cudaMemcpyDeviceToHost);

    // 最適割り当てを求める
    std::vector<int> assignments;
    hungarianAlgorithmInstance.findOptimalAssignment(h_newMatrix, newSize, newSize, assignments);

    // 割り当て結果を表示
    std::cout << "最適な割り当て結果 (行 -> 列):\n";
    for (int row = 0; row < originalRows; ++row) {
        if (assignments[row] != -1) {
            std::cout << "行 " << row << " -> 列 " << assignments[row] << "\n";
        }
    }

    // メモリの解放
    delete[] h_newMatrix;
    cudaFree(d_matrix);
    cudaFree(d_minRow);
    cudaFree(d_minCol);

    return 0;
}
