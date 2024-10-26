#include "GeneticAlgorithm.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <limits>

// コンストラクタの実装
GeneticAlgorithm::GeneticAlgorithm(int population_size, int generations, double mutation_rate, 
                                   int n, const std::vector<std::vector<float>>& cost_matrix)
    : population_size(population_size), generations(generations), mutation_rate(mutation_rate), 
      n(n), cost_matrix(cost_matrix), best_fitness(std::numeric_limits<float>::max()) {}

// 適応度を計算する
float GeneticAlgorithm::fitness(const Individual &individual) const {
    float total_cost = 0.0f;
    for (int i = 0; i < n; ++i) {
        total_cost += cost_matrix[i][individual[i]];
    }
    return total_cost;
}

// ドラフト制度による初期解生成
std::vector<std::vector<int>> GeneticAlgorithm::draft_initial_solutions(const std::vector<std::vector<float>>& cost_matrix, int n, int max_rounds, int num_solutions) {
    std::vector<std::vector<int>> solutions;

    for (int s = 0; s < num_solutions; ++s) {
        std::vector<int> solution(n, -1);  // 最終的なマッチングを格納
        std::vector<bool> assigned_row(n, false);  // 行の割り当て状況を保持
        std::vector<bool> assigned_column(n, false);  // 列の割り当て状況を保持
        std::vector<std::vector<std::pair<int, float>>> row_candidates(n);  // 各行の候補ペア（列とコスト）

        // 各行の最小コスト候補を最大max_roundsまで収集
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<int, float>> candidates;
            for (int j = 0; j < n; ++j) {
                candidates.push_back({j, cost_matrix[i][j]});
            }
            // コストの昇順でソート
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
            // 最大max_roundsの候補を格納
            row_candidates[i].assign(candidates.begin(), candidates.begin() + std::min(max_rounds, (int)candidates.size()));
        }

        // ドラフトステップで優先的にマッチング
        for (int round = 1; round <= max_rounds; ++round) {
            for (int j = 0; j < n; ++j) {  // 列ごとの処理
                if (assigned_column[j]) continue;  // すでにマッチングされた列はスキップ

                int best_row = -1;
                float best_cost = std::numeric_limits<float>::max();

                // 各行を見て、その列がまだマッチングされていない場合、コストを比較
                for (int i = 0; i < n; ++i) {
                    if (!assigned_row[i] && row_candidates[i].size() >= round) {
                        int candidate_column = row_candidates[i][round - 1].first;
                        if (candidate_column == j && row_candidates[i][round - 1].second < best_cost) {
                            best_cost = row_candidates[i][round - 1].second;
                            best_row = i;
                        }
                    }
                }

                // 最良の行をこの列にマッチング
                if (best_row != -1) {
                    solution[best_row] = j;
                    assigned_row[best_row] = true;
                    assigned_column[j] = true;
                }
            }
        }

        // 残りの未割り当ての行をランダムオフセットを付けて貪欲法で埋める
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<> offset_dist(0, n-1);
        int offset = offset_dist(rng);  // ランダムオフセット

        for (int i = 0; i < n; ++i) {
            if (solution[i] == -1) {
                // ランダムオフセットを考慮して、割り当てられていない列を見つけて割り当て
                for (int j = 0; j < n; ++j) {
                    int candidate_column = (j + offset) % n;
                    if (!assigned_column[candidate_column]) {
                        solution[i] = candidate_column;
                        assigned_column[candidate_column] = true;
                        break;
                    }
                }
            }
        }

        // 解を保存
        solutions.push_back(solution);
    }

    return solutions;
}


// 貪欲法による初期解生成
std::vector<int> GeneticAlgorithm::greedy_initial_solution(const std::vector<std::vector<float>>& cost_matrix, int n, int offset) {
    std::vector<int> solution(n, -1); 
    std::vector<bool> assigned(n, false); 

    for (int i = 0; i < n; ++i) {
        int best_j = -1;
        float best_cost = std::numeric_limits<float>::max();
        int row = (i + offset) % n;  

        for (int j = 0; j < n; ++j) {
            if (!assigned[j] && cost_matrix[row][j] < best_cost) {
                best_cost = cost_matrix[row][j];
                best_j = j;
            }
        }

        solution[row] = best_j;
        assigned[best_j] = true;
    }

    return solution;
}

// ランダムな個体を生成
GeneticAlgorithm::Individual GeneticAlgorithm::random_individual() const {
    Individual individual(n);
    std::iota(individual.begin(), individual.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(individual.begin(), individual.end(), g);
    return individual;
}

// ランダムな貪欲法
GeneticAlgorithm::Individual GeneticAlgorithm::random_greedy_individual(const std::vector<std::vector<float>>& cost_matrix) const {
    Individual individual(n);
    Individual solution(n); 
    std::iota(individual.begin(), individual.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(individual.begin(), individual.end(), g);
    std::vector<bool> assigned(n, false); 

    for (int i = 0; i < n; ++i) {
        int best_j = -1;
        float best_cost = std::numeric_limits<float>::max();
        int row = individual[i];  

        for (int j = 0; j < n; ++j) {
            if (!assigned[j] && cost_matrix[row][j] < best_cost) {
                best_cost = cost_matrix[row][j];
                best_j = j;
            }
        }

        solution[row] = best_j;
        assigned[best_j] = true;
    }

    return solution;
}

// 部分写像交叉（PMX）
std::vector<int> GeneticAlgorithm::pmx_crossover(const std::vector<int>& parent1, const std::vector<int>& parent2, std::mt19937& gen) {
    std::uniform_int_distribution<> dist(0, n-1);
    int start = dist(gen);
    int end = dist(gen);

    if (start > end) std::swap(start, end);

    std::vector<int> child(n, -1);
    std::vector<bool> used(n, false);

    // 親1から部分コピー
    for (int i = start; i <= end; ++i) {
        child[i] = parent1[i];
        used[child[i]] = true;  // 使用済みとしてマーク
    }

    // 親2からの残りを順番に埋める
    for (int i = 0; i < n; ++i) {
        if (child[i] == -1) {  // 未確定部分を埋める
            for (int j = 0; j < n; ++j) {
                if (!used[parent2[j]]) {  // 親2の要素で未使用のものを見つける
                    child[i] = parent2[j];
                    used[child[i]] = true;  // 使用済みマーク
                    break;
                }
            }
        }
    }

    return child;
}

// 突然変異
void GeneticAlgorithm::mutate(std::vector<int>& individual, std::mt19937& gen) const {
    std::uniform_int_distribution<int> index_dist(0, n-1);
    int idx1 = index_dist(gen);
    int idx2 = index_dist(gen);
    std::swap(individual[idx1], individual[idx2]);
}

// トーナメント選択
std::vector<int> GeneticAlgorithm::tournament_selection(const std::vector<Individual>& population, const std::vector<float>& fitness, int tournament_size, std::mt19937& gen) const {
    std::uniform_int_distribution<> dis(0, population_size-1);
    int best_index = dis(gen);

    for (int i = 1; i < tournament_size; ++i) {
        int competitor_index = dis(gen);
        if (fitness[competitor_index] < fitness[best_index]) {
            best_index = competitor_index;
        }
    }

    return population[best_index];
}

// 最適化実行
std::vector<int> GeneticAlgorithm::optimize() {
    std::mt19937 gen(std::random_device{}());

    // 個体群の初期化
    std::vector<std::vector<int>> population(population_size);
    
    // ドラフト制度による初期解生成
    // int draftSize = std::min(population_size / 2, (n / 2) );
    // std::vector<std::vector<int>> draft_solutions = draft_initial_solutions(cost_matrix, n, 15, draftSize);
    // for (int i = 0; i < draftSize; ++i) {
    //     population[i] = draft_solutions[i];
    //     std::cout << "Draft solution " << i << " fitness: " << fitness(population[i]) << std::endl;
    // }

    // 貪欲法による初期解生成
    // int greedySize = std::min(population_size - draftSize, n);
    // for (int i = draftSize; i < greedySize + draftSize; ++i) {
    //     population[i] = greedy_initial_solution(cost_matrix, n, i);
    //     std::cout << "Greedy solution with offset " << i << " fitness: " << fitness(population[i]) << std::endl;
    // }

    // 残りはランダム貪欲法による初期解生成
    int greedySize = 0;
    int draftSize = 0;
    for (int i = greedySize + draftSize; i < population_size; ++i) {
        population[i] = random_greedy_individual(cost_matrix);
    }

    std::vector<float> fitness(population_size);
    std::vector<std::vector<int>> new_population(population_size);

    // 世代ごとの最適化
    for (int generation = 0; generation < generations; ++generation) {
        // 適応度の計算
        for (int i = 0; i < population_size; ++i) {
            fitness[i] = this->fitness(population[i]);
        }

        // 新しい個体群の生成
        for (int i = 0; i < population_size; ++i) {
            // 親の選択
            std::vector<int> parent1 = tournament_selection(population, fitness, 7, gen);
            std::vector<int> parent2 = tournament_selection(population, fitness, 7, gen);

            // 交叉
            std::vector<int> child = pmx_crossover(parent1, parent2, gen);

            // 突然変異
            std::uniform_real_distribution<double> mutation_dist(0.0, 1.0);
            if (mutation_dist(gen) < mutation_rate) {
                mutate(child, gen);
            }

            new_population[i] = child;
        }

        // 新しい個体群に置き換え
        population = new_population;

        // 最良の個体を記録
        for (const auto& individual : population) {
            float current_fitness = this->fitness(individual);
            if (current_fitness < best_fitness) {
                best_fitness = current_fitness;
                best_individual = individual;
            }
        }

        if (generation % 100 == 0) {
            std::cout << "Generation " << generation << " - Best Fitness: " << best_fitness << std::endl;
        }
    }

    return best_individual;
}

// 最小コストの取得
float GeneticAlgorithm::getBestCost() const {
    return best_fitness;
}
