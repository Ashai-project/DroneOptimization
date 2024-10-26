#include "hungarian_algorithm_para.h"

HungarianAlgorithmPARA::HungarianAlgorithmPARA() {}

// 行列から最適な割り当てを見つける
std::vector<int> HungarianAlgorithmPARA::findOptimalAssignment(const std::vector<std::vector<float>>& costMatrix) {
    int rows = costMatrix.size();
    int cols = costMatrix[0].size();
    int dim = std::max(rows, cols);

    // コスト行列の正方行列化（ダミー要素を追加）
    std::vector<std::vector<float>> cost(dim, std::vector<float>(dim, 0.0f));

    // 最上位のforループを並列化
    #pragma omp parallel for collapse(1) num_threads(4)
    for (int row = 0; row < dim; ++row) {
        for (int col = 0; col < dim; ++col) {
            if (row < rows && col < cols) {
                cost[row][col] = costMatrix[row][col];
            } else {
                cost[row][col] = 0.0f;
            }
        }
    }

    // ハンガリアンアルゴリズムの実装
    std::vector<float> potentialRow(dim + 1, 0.0f); // 行のポテンシャル
    std::vector<float> potentialCol(dim + 1, 0.0f); // 列のポテンシャル
    std::vector<float> minValues(dim + 1);           // 最小値を保持
    std::vector<int> matching(dim + 1, 0);           // マッチング情報
    std::vector<int> path(dim + 1, 0);              // 経路情報

    // 各行に対して最適な割り当てを見つける
    for (int row = 1; row <= dim; ++row) {
        matching[0] = row;
        int freeCol = 0;
        std::fill(minValues.begin(), minValues.end(), std::numeric_limits<float>::max());
        std::vector<bool> visited(dim + 1, false);

        // 最短増加パスを見つける
        while (matching[freeCol] != 0) {
            visited[freeCol] = true; // 現在の列を訪問済みに設定
            int currentRow = matching[freeCol]; // 現在の行を取得
            float delta = std::numeric_limits<float>::max();
            int nextCol = 0;

            // 未訪問の各列に対して、最小の調整済みコストを探す
            // #pragma omp parallel for num_threads(2)
            for (int col = 1; col <= dim; ++col) {
                if (!visited[col]) {
                    // 調整済みコストを計算
                    float currentCost = cost[currentRow - 1][col - 1] - potentialRow[currentRow] - potentialCol[col];
                    // #pragma omp critical
                    {
                        if (currentCost < minValues[col]) {
                            minValues[col] = currentCost; // 最小値を更新
                            path[col] = freeCol;          // 経路を記録
                        }
                        if (minValues[col] < delta) {
                            delta = minValues[col];       // deltaを更新
                            nextCol = col;                // 次の列を設定
                        }
                    }
                }
            }

            // ポテンシャルを更新
            // #pragma omp parallel for num_threads(2)
            for (int col = 0; col <= dim; ++col) {
                if (visited[col]) {
                    potentialRow[matching[col]] += delta;
                    potentialCol[col] -= delta;
                } else {
                    minValues[col] -= delta;
                }
            }

            freeCol = nextCol; // 次の列に移動
        }

        // 増強パスに沿ってマッチングを更新
        while (freeCol != 0) {
            int prevCol = path[freeCol];            // 前の列を取得
            matching[freeCol] = matching[prevCol];  // マッチングを更新
            freeCol = prevCol;                      // フリーな列を更新
        }
    }

    std::vector<int> assignment(rows, -1);
    // #pragma omp parallel for num_threads(2)
    for (int col = 1; col <= dim; ++col) {
        if (matching[col] <= rows && col <= cols) {
            assignment[matching[col] - 1] = col - 1;
        }
    }

    // 重複がないかチェック（必要に応じて）
    if (checkForDuplicates(assignment)) {
        std::cerr << "Error: Assignment has duplicates." << std::endl;
    }

    return assignment;
}

bool HungarianAlgorithmPARA::checkForDuplicates(const std::vector<int>& assignment) {
    std::set<int> assignedJobs;
    for (int job : assignment) {
        if (job != -1) {
            if (assignedJobs.find(job) != assignedJobs.end()) {
                return true;  // 重複あり
            }
            assignedJobs.insert(job);
        }
    }
    return false;  // 重複なし
}

float HungarianAlgorithmPARA::calculateCost(const std::vector<int>& assignment, const std::vector<std::vector<float>>& costMatrix) {
    float totalCost = 0.0f;
    int rows = costMatrix.size();

    if (rows == 0) {
        std::cerr << "Error: Cost matrix has no rows." << std::endl;
        return 0.0f;
    }

    int cols = costMatrix[0].size();
    if (cols == 0) {
        std::cerr << "Error: Cost matrix has no columns." << std::endl;
        return 0.0f;
    }

    // #pragma omp parallel for reduction(+:totalCost)
    for (int i = 0; i < assignment.size(); ++i) {
        if (assignment[i] != -1) {
            if (i >= rows) {
                std::cerr << "Error: Row index out of bounds at index " << i << std::endl;
                continue;
            }
            if (assignment[i] >= cols) {
                std::cerr << "Error: Column index out of bounds at assignment[" << i << "] = " << assignment[i] << std::endl;
                continue;
            }
            totalCost += costMatrix[i][assignment[i]];
        }
    }
    return totalCost;
}
