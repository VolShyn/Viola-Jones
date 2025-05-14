#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "Image.h"
#include "HaarFeature.h"
#include "AdaBoost.h"
#include "CascadeClassifier.h"

#include <vector>
#include <cstddef>
#include <functional>

namespace vj {

struct TrainerOptions {
    std::size_t window_size = 24;
    std::size_t num_rounds   = 10;   // weak learners per stage
    double      target_FPR   = 0.5;  // false-positive rate per stage
    double      target_TPR   = 0.99; // detection rate per stage
};

// progress callback type for tracking training progress
using ProgressCallback = std::function<void(int currentStage, int totalStages,
                                           int currentRound, int totalRounds)>;

// found this cool thing with  multi-string comments
/**
 * little utility that:
 *   1) enumerates all two-rectangle Haar features in an N×N window,
 *   2) trains an AdaBoost strong classifier
 *   3) repeats to build a cascade of stages
 */
class Trainer {
public:
    /**
     * train a single AdaBoost stage
     * @param  posIs       integral‐images of positive windows
     * @param  negIs       integral‐images of negative windows
     * @param  opts        controls #rounds, etc
     * @return             a trained AdaBoost strong classifier
     */
    static AdaBoost<int>
    trainStage(const std::vector<Image<long long>>& posIs,
               const std::vector<Image<long long>>& negIs,
               const TrainerOptions& opts);

    /**
     * buidling a cascade by repeatedly calling trainStage
     * dropping "easy negatives" at each step
     */
    static CascadeClassifier<int>
    trainCascade(std::vector<Image<long long>> posIs,
                 std::vector<Image<long long>> negIs,
                 const TrainerOptions& opts);

    /**
     * building a cascade with progress tracking
     * @param  posIs             integral‐images of positive windows
     * @param  negIs             integral‐images of negative windows
     * @param  opts              controls #rounds, etc
     * @param  progressCallback  callback function for progress updates
     * @return                   a trained cascade classifier
     */
    static CascadeClassifier<int>
    trainCascade(std::vector<Image<long long>> posIs,
                 std::vector<Image<long long>> negIs,
                 const TrainerOptions& opts,
                 ProgressCallback progressCallback);
};

} // namespace vj

#endif // TRAINER_HPP
