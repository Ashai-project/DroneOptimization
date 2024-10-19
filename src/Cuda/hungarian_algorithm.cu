#include "hungarian_algorithm.h"

// コンストラクタ
HungarianAlgorithm::HungarianAlgorithm() {
    cudaMalloc(&d_minVal, sizeof(float));  // デバイスメモリを初期化
}

// デストラクタ
HungarianAlgorithm::~HungarianAlgorithm() {
    cudaFree(d_minVal);  // デバイスメモリの解放
}

// ダミー行または列を追加して正方行列を作成
void HungarianAlgorithm::addDummyElements(float *matrix, int rows, int cols, int newSize) {
    for (int i = rows; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            matrix[i * newSize + j] = FLT_MAX;  // ダミー行に非常に大きなコストを設定
        }
    }

    for (int i = 0; i < newSize; ++i) {
        for (int j = cols; j < newSize; ++j) {
            matrix[i * newSize + j] = FLT_MAX;  // ダミー列に非常に大きなコストを設定
        }
    }
}

// CUDA カーネル：行ごとの最小値を計算し、全要素から引く
__global__ void subtractRowMin(float *matrix, float *minRow, int rows, int cols) {
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

// CUDA カーネル：列ごとの最小値を計算し、全要素から引く
__global__ void subtractColMin(float *matrix, float *minCol, int rows, int cols) {
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

// カバーされていない行の最小値を探すカーネル
__global__ void findMinVal(float *matrix, int *rowCovered, int *colCovered, float *minVal, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || rowCovered[row]) return;

    float localMin = FLT_MAX;
    for (int col = 0; col < cols; ++col) {
        if (!colCovered[col] && matrix[row * cols + col] < localMin) {
            localMin = matrix[row * cols + col];
        }
    }

    // スレッドごとの最小値をアトミックに更新
    atomicMin((int*)minVal, __float_as_int(localMin));
}

// カバーされていない行から最小値を引くカーネル
__global__ void subtractMinFromRows(float *matrix, int *rowCovered, float minVal, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || rowCovered[row]) return;

    for (int col = 0; col < cols; ++col) {
        matrix[row * cols + col] -= minVal;
    }
}

// カバーされている列に最小値を加えるカーネル
__global__ void addMinToCols(float *matrix, int *colCovered, float minVal, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols || !colCovered[col]) return;

    for (int row = 0; row < rows; ++row) {
        matrix[row * cols + col] += minVal;
    }
}

// カバーを更新する補正操作
void HungarianAlgorithm::updateMatrixForUncovered(float *matrix, int *rowCovered, int *colCovered, int rows, int cols) {
    float h_minVal = FLT_MAX;  // ホスト側で FLT_MAX を設定

    // `cudaMemcpy` を使って FLT_MAX をデバイスメモリにコピー
    cudaMemcpy(d_minVal, &h_minVal, sizeof(float), cudaMemcpyHostToDevice);

    // カバーされていない行の最小値を探す
    findMinVal<<<rows, 1>>>(matrix, rowCovered, colCovered, d_minVal, rows, cols);
    cudaDeviceSynchronize();

    // 最小値をカバーされていない行から引く
    float minVal;
    cudaMemcpy(&minVal, d_minVal, sizeof(float), cudaMemcpyDeviceToHost);
    subtractMinFromRows<<<rows, 1>>>(matrix, rowCovered, minVal, rows, cols);
    cudaDeviceSynchronize();

    // カバーされている列に最小値を加える
    addMinToCols<<<cols, 1>>>(matrix, colCovered, minVal, rows, cols);
    cudaDeviceSynchronize();
}


// 最適割り当てを計算するメソッド
void HungarianAlgorithm::findOptimalAssignment(float *matrix, int rows, int cols, std::vector<int>& assignments) {
    std::vector<int> rowCovered(rows, 0);
    std::vector<int> colCovered(cols, 0);

    assignments.resize(rows, -1);

    bool done = false;
    while (!done) {
        done = true;
        for (int row = 0; row < rows; ++row) {
            if (assignments[row] == -1) {
                done = false;
                for (int col = 0; col < cols; ++col) {
                    if (matrix[row * cols + col] == 0 && !rowCovered[row] && !colCovered[col]) {
                        assignments[row] = col;
                        rowCovered[row] = 1;
                        colCovered[col] = 1;
                        break;
                    }
                }
            }
        }

        if (!done) {
            updateMatrixForUncovered(matrix, rowCovered.data(), colCovered.data(), rows, cols);
        }
    }
}

// ホストからカーネルを実行するラッパーメソッド
void HungarianAlgorithm::runHungarianAlgorithm(float *d_matrix, float *d_minRow, float *d_minCol, int size) {
    subtractRowMin<<<size, 1>>>(d_matrix, d_minRow, size, size);
    cudaDeviceSynchronize();

    subtractColMin<<<size, 1>>>(d_matrix, d_minCol, size, size);
    cudaDeviceSynchronize();
}
