#include "GeneticAlgorithm.h"

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

// 部分写像交叉（PMX）に置き換え
std::vector<GeneticAlgorithm::Individual> GeneticAlgorithm::pmx_crossover(const std::vector<Individual>& parents) {
    std::vector<Individual> offspring;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, n-1);

    // 各親のカップルに対してPMX交叉を行う
    for (size_t i = 0; i < population_size; i += 2) {
        const Individual &parent1 = parents[i];
        const Individual &parent2 = parents[i + 1];
        Individual child1(n, -1), child2(n, -1);

        // 交叉範囲をランダムに選択
        int start = dist(gen);
        int end = dist(gen);
        if (start > end) std::swap(start, end);

        // 部分配列をコピー
        for (int j = start; j <= end; ++j) {
            child1[j] = parent1[j];
            child2[j] = parent2[j];
        }

        // 残りの要素を親から順番に埋める
        auto fill_remaining = [&](Individual &child, const Individual &parent) {
            int current = (end + 1) % n;
            for (int j = 0; j < n; ++j) {
                int idx = (end + 1 + j) % n;
                if (std::find(child.begin(), child.end(), parent[idx]) == child.end()) {
                    child[current] = parent[idx];
                    current = (current + 1) % n;
                }
            }
        };

        fill_remaining(child1, parent2);
        fill_remaining(child2, parent1);

        offspring.push_back(child1);
        offspring.push_back(child2);
    }

    return offspring;
}

// 突然変異
void GeneticAlgorithm::mutate(Individual &individual) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    if (dis(gen) < mutation_rate) {
        std::uniform_int_distribution<> index_dist(0, n-1);
        int idx1 = index_dist(gen);
        int idx2 = index_dist(gen);
        std::swap(individual[idx1], individual[idx2]);

        // 複数の要素を入れ替える処理
        if (dis(gen) < 0.5) {
            int idx3 = index_dist(gen);
            std::swap(individual[idx1], individual[idx3]);
        }
    }
}

// トーナメント選択
GeneticAlgorithm::Individual GeneticAlgorithm::tournament_selection(const std::vector<Individual> &population) const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, population_size-1);

    const int TOURNAMENT_SIZE = 7;
    Individual best_individual = population[dis(gen)];
    for (int i = 1; i < TOURNAMENT_SIZE; ++i) {
        Individual contender = population[dis(gen)];
        if (fitness(contender) < fitness(best_individual)) {
            best_individual = contender;
        }
    }

    return best_individual;
}

// JGGによる世代交代
void GeneticAlgorithm::jgg_generation(std::vector<Individual>& population, const std::vector<Individual>& offspring) {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t i = 0; i < offspring.size(); ++i) {
        std::uniform_int_distribution<> dis(0, population_size-1);
        population[dis(gen)] = offspring[i];
    }
}

// 最適化実行
std::vector<int> GeneticAlgorithm::optimize() {
    std::vector<Individual> population;

    // 個体群の初期化
    for (int offset = 0; offset < population_size; ++offset) {
        Individual initial_solution = greedy_initial_solution(cost_matrix, n, offset);
        population.push_back(initial_solution);
        std::cout << "Greedy solution with offset " << offset << " fitness: " << fitness(initial_solution) << std::endl;
    }

    // for (int i = 5; i < population_size; ++i) {
    //     population.push_back(random_individual());
    // }

    for (int generation = 0; generation < generations; ++generation) {
        // トーナメント選択で親個体を選択
        std::vector<Individual> parents;
        for (int i = 0; i < population_size; ++i) {
            parents.push_back(tournament_selection(population));
        }

        // PMXによる交叉
        std::vector<Individual> offspring = pmx_crossover(parents);

        // 突然変異を適用
        for (Individual& child : offspring) {
            mutate(child);
        }

        // JGGによる世代交代
        jgg_generation(population, offspring);

        // 最良の個体を記録
        for (const Individual &individual : population) {
            float current_fitness = fitness(individual);
            if (current_fitness < best_fitness) {
                best_fitness = current_fitness;
                best_individual = individual;
            }
        }

        std::cout << "Generation " << generation << " - Best Fitness: " << best_fitness << std::endl;
    }

    return best_individual;
}

// 最小コストの取得
float GeneticAlgorithm::getBestCost() const {
    return best_fitness;
}
