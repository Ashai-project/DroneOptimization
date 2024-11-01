#ifndef HUNGARIAN_ALGORITHM_CPU_H
#define HUNGARIAN_ALGORITHM_CPU_H

#include <vector>

class HungarianAlgorithmCPU {
public:
    HungarianAlgorithmCPU();
    std::vector<int> findOptimalAssignment(const std::vector<std::vector<float>>& costMatrix);
    float calculateCost(const std::vector<int>& assignment, const std::vector<std::vector<float>>& costMatrix);

private:
    bool checkForDuplicates(const std::vector<int>& assignment);
};

#endif // HUNGARIAN_ALGORITHM_CPU_H
