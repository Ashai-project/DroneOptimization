#include "hungarian_algorithm.h"

// CUDAカーネル定義
__global__ void subtractRowMinKernel(float* matrix, float* minRow, int rows, int cols);
__global__ void subtractColMinKernel(float* matrix, float* minCol, int rows, int cols);
__global__ void findMinValKernel(float* matrix, int* rowCovered, int* colCovered, float* minVal, int rows, int cols);

// コンストラクタ
HungarianAlgorithm::HungarianAlgorithm() {
    cudaMalloc(&d_minVal, sizeof(float));
}

// デストラクタ
HungarianAlgorithm::~HungarianAlgorithm() {
    cudaFree(d_minVal);
}

void HungarianAlgorithm::addDummyElements(float* matrix, int rows, int cols, int newSize) {
    for (int i = rows; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            matrix[i * newSize + j] = FLT_MAX;
        }
    }
    for (int i = 0; i < newSize; ++i) {
        for (int j = cols; j < newSize; ++j) {
            matrix[i * newSize + j] = FLT_MAX;
        }
    }
}

// 最適な割り当てを見つける
std::vector<int> HungarianAlgorithm::findOptimalAssignment(const std::vector<std::vector<float>>& costMatrix) {
    int rows = costMatrix.size();
    int cols = costMatrix[0].size();
    int newSize = std::max(rows, cols);

    // 正方行列化（1次元配列に変換）
    std::vector<float> matrix(newSize * newSize, std::numeric_limits<float>::max());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * newSize + j] = costMatrix[i][j];
        }
    }

    // GPUメモリにコスト行列を転送
    cudaMalloc(&d_matrix, newSize * newSize * sizeof(float));
    cudaMemcpy(d_matrix, matrix.data(), newSize * newSize * sizeof(float), cudaMemcpyHostToDevice);

    // 行の最小値を引く
    cudaMalloc(&d_minRow, newSize * sizeof(float));
    subtractRowMinKernel<<<newSize, 1>>>(d_matrix, d_minRow, newSize, newSize);
    cudaDeviceSynchronize();

    // 列の最小値を引く
    cudaMalloc(&d_minCol, newSize * sizeof(float));
    subtractColMinKernel<<<newSize, 1>>>(d_matrix, d_minCol, newSize, newSize);
    cudaDeviceSynchronize();

    // 行列をホストに戻す
    cudaMemcpy(matrix.data(), d_matrix, newSize * newSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<int> rowAssignment(newSize, -1);
    std::vector<int> rowCovered(newSize, 0);  // 0: 未カバー, 1: カバー済み
    std::vector<int> colCovered(newSize, 0);  // 0: 未カバー, 1: カバー済み

    bool optimalFound = false;
    while (!optimalFound) {
        bool newAssignment = false;

        for (int i = 0; i < newSize; ++i) {
            if (!rowCovered[i]) {
                for (int j = 0; j < newSize; ++j) {
                    if (matrix[i * newSize + j] == 0 && !colCovered[j]) {
                        rowAssignment[i] = j;
                        rowCovered[i] = 1;
                        colCovered[j] = 1;
                        newAssignment = true;
                        break;
                    }
                }
            }
        }

        if (std::count(rowAssignment.begin(), rowAssignment.end(), -1) == 0) {
            optimalFound = true;
        }

        if (!newAssignment) {
            // カバーされていない行と列の最小値を探す
            float minVal = std::numeric_limits<float>::max();
            for (int i = 0; i < newSize; ++i) {
                if (!rowCovered[i]) {  // カバーされていない行
                    for (int j = 0; j < newSize; ++j) {
                        if (!colCovered[j]) {  // カバーされていない列
                            minVal = std::min(minVal, matrix[i * newSize + j]);
                        }
                    }
                }
            }

            // カバーされていない行から最小値を引き、カバーされている列に最小値を加える
            for (int i = 0; i < newSize; ++i) {
                if (!rowCovered[i]) {
                    for (int j = 0; j < newSize; ++j) {
                        matrix[i * newSize + j] -= minVal;
                    }
                }
            }
            for (int j = 0; j < newSize; ++j) {
                if (colCovered[j]) {
                    for (int i = 0; i < newSize; ++i) {
                        matrix[i * newSize + j] += minVal;
                    }
                }
            }

            // カバー解除を行い、再度探索
            std::fill(rowCovered.begin(), rowCovered.end(), 0);
            std::fill(colCovered.begin(), colCovered.end(), 0);
        }
    }

    // メモリ解放
    cudaFree(d_matrix);
    cudaFree(d_minRow);
    cudaFree(d_minCol);

    return rowAssignment;
}

// コストを計算する
float HungarianAlgorithm::calculateCost(const std::vector<int>& assignment, const std::vector<std::vector<float>>& costMatrix) {
    float totalCost = 0.0f;
    for (int i = 0; i < assignment.size(); ++i) {
        if (assignment[i] != -1 && i < costMatrix.size() && assignment[i] < costMatrix[0].size()) {
            totalCost += costMatrix[i][assignment[i]];
        }
    }
    return totalCost;
}

// CUDAカーネルの実装
__global__ void subtractRowMinKernel(float* matrix, float* minRow, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float minVal = matrix[row * cols];
        for (int col = 1; col < cols; ++col) {
            if (matrix[row * cols + col] < minVal) {
                minVal = matrix[row * cols + col];
            }
        }
        for (int col = 0; col < cols; ++col) {
            matrix[row * cols + col] -= minVal;
        }
        minRow[row] = minVal;
    }
}

__global__ void subtractColMinKernel(float* matrix, float* minCol, int rows, int cols) {
    int col = blockIdx.x;
    if (col < cols) {
        float minVal = matrix[col];
        for (int row = 1; row < rows; ++row) {
            if (matrix[row * cols + col] < minVal) {
                minVal = matrix[row * cols + col];
            }
        }
        for (int row = 0; row < rows; ++row) {
            matrix[row * cols + col] -= minVal;
        }
        minCol[col] = minVal;
    }
}
