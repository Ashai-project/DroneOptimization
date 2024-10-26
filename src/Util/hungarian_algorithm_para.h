#ifndef HUNGARIAN_ALGORITHM_PARA_H
#define HUNGARIAN_ALGORITHM_PARA_H

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <set>
#include <omp.h>

class HungarianAlgorithmPARA {
public:
    HungarianAlgorithmPARA();
    std::vector<int> findOptimalAssignment(const std::vector<std::vector<float>>& costMatrix);
    float calculateCost(const std::vector<int>& assignment, const std::vector<std::vector<float>>& costMatrix);

private:
    bool checkForDuplicates(const std::vector<int>& assignment);
};

#endif // HUNGARIAN_ALGORITHM_PARA_H
