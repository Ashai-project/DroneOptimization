#ifndef GENETIC_ALGORITHM_PARA_H
#define GENETIC_ALGORITHM_PARA_H

#include <iostream>
#include <numeric>
#include <limits>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <cuda_runtime.h>
#include <curand_kernel.h>

class GeneticAlgorithmPARA {
private:
    using Individual = std::vector<int>;

public:
    // コンストラクタで設定を受け取る
    GeneticAlgorithmPARA(int population_size, int generations, double mutation_rate, 
                     int n, const std::vector<float>& cost_matrix);

    // 最適化実行関数
    Individual optimize();
    
    // 最小コストの取得
    float getBestCost() const;

private:
    int population_size;     // 個体群のサイズ
    int generations;         // 世代数
    double mutation_rate;     // 突然変異率
    int n;                   // 問題のサイズ（行列のサイズ）
    std::vector<float> cost_matrix;  // コスト行列
    Individual best_individual;  // 最良の割り当て
    float best_fitness;        // 最小コスト

    // 適応度を計算する関数
    float fitness(const Individual &individual) const;

    std::vector<Individual> draft_initial_solutions(const std::vector<float>& cost_matrix, int n, int max_rounds, int num_solutions);

    Individual greedy_initial_solution(const std::vector<float>& cost_matrix, int n, int offset);

    // 個体をランダムに生成
    Individual random_individual() const;

    Individual random_greedy_individual(const std::vector<float>& cost_matrix) const;
};

#endif // GENETIC_ALGORITHM_PARA_H
