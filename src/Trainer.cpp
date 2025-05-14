#include "viola_jones/Trainer.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <iostream>
#include <numeric>
#include <functional>
#include <iomanip>

namespace vj {

// callback type for progress reporting
using ProgressCallback = std::function<void(int currentStage, int totalStages,
                                           int currentRound, int totalRounds)>;

// here we generate every two-rectangle Haar feature in a window_size×window_size region
// featyres include horizontal, vertical, and diagonal two-rectangle features
// as in their paper
static
std::vector<HaarFeature<int>>
makeAllHaarFeatures(std::size_t window_size)
{
    std::vector<HaarFeature<int>> feats;
    // horizontal two-rectangles (side by side)
    for (std::size_t w = 1; w <= window_size; ++w) {
      for (std::size_t h = 1; h*2 <= window_size; ++h) {
        for (std::size_t x = 0; x + w <= window_size; ++x) {
          for (std::size_t y = 0; y + 2*h <= window_size; ++y) {
            Rect<int> white{ static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h) };
            Rect<int> black{ static_cast<int>(x), static_cast<int>(y+h), static_cast<int>(w), static_cast<int>(h) };
            feats.emplace_back(white, black);
          }
        }
      }
    }
    // and vertical two-rectangles (one above the other)
    for (std::size_t h = 1; h <= window_size; ++h) {
      for (std::size_t w = 1; w*2 <= window_size; ++w) {
        for (std::size_t y = 0; y + h <= window_size; ++y) {
          for (std::size_t x = 0; x + 2*w <= window_size; ++x) {
            Rect<int> white{ static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h) };
            Rect<int> black{ static_cast<int>(x+w), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h) };
            feats.emplace_back(white, black);
          }
        }
      }
    }

    std::cout << "Generated " << feats.size() << " Haar features for window size " << window_size << "x" << window_size << std::endl;
    std::cout.flush();
    return feats;
}




// train a single AdaBoost stage
AdaBoost<int>
Trainer::trainStage(
    const std::vector<Image<long long>>& posIs,
    const std::vector<Image<long long>>& negIs,
    const TrainerOptions& opts)
{
    std::size_t Npos = posIs.size(), Nneg = negIs.size();
    std::size_t N = Npos + Nneg;
    // init weights
    std::vector<double> w(N);
    double w0 = 1.0 / (2*Npos), w1 = 1.0 / (2*Nneg);
    for (std::size_t i = 0; i < N; ++i)
      w[i] = (i < Npos ? w0 : w1);

    // collect all features
    auto allFeats = makeAllHaarFeatures(opts.window_size);

    AdaBoost<int> strong;
    double sumAlphas = 0;

    // run for R rounds
    for (std::size_t r = 0; r < opts.num_rounds; ++r) {
      // 1) find best weak: feature + threshold + polarity minimizing weighted error
      double bestErr = std::numeric_limits<double>::infinity();
      typename AdaBoost<int>::Weak bestW{ allFeats[0], 0, 1, 0 };

      for (auto const& feat : allFeats) {
        // evaluate feature on all samples
        std::vector<double> vals(N);
        for (std::size_t i = 0; i < Npos; ++i) vals[i] = feat(posIs[i], 0, 0);
        for (std::size_t i = 0; i < Nneg; ++i) vals[Npos + i] = feat(negIs[i], 0, 0);

        // sort to scan thresholds
        std::vector<std::size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
          [&](auto a, auto b){ return vals[a] < vals[b]; });

        // try thresholds between distinct vals
        double sumPos = 0, sumNeg = 0;
        for (auto wi : w) {
          // nothing here – we'll recompute errors below
        }

        // lets keep a running error for polarity +1 and -1
        double wPos=0, wNeg=0;
        for (auto i : idx) {
          bool label = (i < Npos);
          double weight = w[i];
          // consider threshold at vals[i]
          // for polarity = +1: predict positive if val < thresh
          // error = sum weights where pred != label
          // We can update wPos, wNeg incrementally, but for clarity:
          double thr = vals[i];
          for (int polarity : {+1, -1}) {
            double err = 0;
            for (std::size_t k = 0; k < N; ++k) {
              bool pred = (polarity * vals[k] < polarity * thr);
              if (pred != (k < Npos)) err += w[k];
            }
            if (err < bestErr) {
              bestErr = err;
              bestW = { feat, static_cast<int>(thr), polarity, 0.0 };
            }
          }
        }
      }

      // 2) compute alpha and add weak
      double err = std::max(bestErr, 1e-10);
      double alpha = 0.5 * std::log((1 - err) / err);
      bestW.alpha = alpha;
      strong.add(std::move(bestW));
      sumAlphas += alpha;

      // 3) update weights
      for (std::size_t i = 0; i < N; ++i) {
        auto& weight = w[i];
        bool label = (i < Npos);
        // re-evaluate weak on sample i
        auto& wkl = strong.weaks_.back();  // last weak
        bool pred = (wkl.polarity *
                     (i < Npos ? wkl.feat(posIs[i],0,0)
                               : wkl.feat(negIs[i-Npos],0,0))
                    < wkl.polarity * wkl.thresh);
        weight *= std::exp(-alpha * (label ? +1 : -1) * (pred ? +1 : -1));
      }
      // normalize
      double Z = std::accumulate(w.begin(), w.end(), 0.0);
      for (auto& weight : w) weight /= Z;
    }

    // 4) set the strong threshold to half the total alpha
    strong.setThreshold(0.5 * sumAlphas);
    return strong;
}

// 3) train a cascade by chaining multiple stages, each time removing true negatives.
CascadeClassifier<int>
Trainer::trainCascade(
    std::vector<Image<long long>> posIs,
    std::vector<Image<long long>> negIs,
    const TrainerOptions& opts,
    ProgressCallback progressCallback)
{
    CascadeClassifier<int> cascade;
    double overallFPR = 1.0;
    size_t initial_neg_count = negIs.size();

    std::cout << "Starting cascade training with " << posIs.size() << " positive and "
              << negIs.size() << " negative samples" << std::endl;

    // estimate the number of stages needed (for progress tracking)
    int estimatedTotalStages = 10; // arbitrary estimate
    int currentStage = 0;

    while (overallFPR > 0.01) {  // stop when cascade is good enough
      currentStage++;

      // train stage with progress tracking for each round
      AdaBoost<int> stage;

      // prep for training a stage
      std::size_t Npos = posIs.size(), Nneg = negIs.size();
      std::size_t N = Npos + Nneg;
      // init weights
      std::vector<double> w(N);
      double w0 = 1.0 / (2*Npos), w1 = 1.0 / (2*Nneg);
      for (std::size_t i = 0; i < N; ++i)
        w[i] = (i < Npos ? w0 : w1);

      // collect all features
      auto allFeats = makeAllHaarFeatures(opts.window_size);

      double sumAlphas = 0;

      // run for R rounds with progress tracking
      for (std::size_t r = 0; r < opts.num_rounds; ++r) {
        // Report progress - include debug output
        std::cout << "Training stage " << currentStage << ", round " << (r+1) << "/" << opts.num_rounds << "..." << std::endl;
        std::cout.flush();
        if (progressCallback) {
          progressCallback(currentStage, estimatedTotalStages, r+1, opts.num_rounds);
        }

        // 1) find best weak: feature + threshold + polarity minimizing weighted error
        double bestErr = std::numeric_limits<double>::infinity();
        typename AdaBoost<int>::Weak bestW{ allFeats[0], 0, 1, 0 };

        std::cout << "Evaluating " << allFeats.size() << " features..." << std::endl;
        std::cout.flush();

        // we limit the number of features for faster execution during testing
        // in their work they have 6000
        // and computers that were x50 slower...
        std::size_t featuresToEvaluate = std::min(allFeats.size(), static_cast<std::size_t>(5000));
        std::cout << "Using " << featuresToEvaluate << " features for this round" << std::endl;
        std::cout.flush();

        for (std::size_t featIndex = 0; featIndex < featuresToEvaluate; ++featIndex) {
          auto const& feat = allFeats[featIndex];

          // show progress periodically
          // I want tqdm for this language
          if (featIndex % 500 == 0) {
            std::cout << "Evaluated " << featIndex << "/" << featuresToEvaluate << " features" << std::endl;
            std::cout.flush();
          }
          // evaluate feature on all samples
          std::vector<double> vals(N);
          for (std::size_t i = 0; i < Npos; ++i) vals[i] = feat(posIs[i], 0, 0);
          for (std::size_t i = 0; i < Nneg; ++i) vals[Npos + i] = feat(negIs[i], 0, 0);

          // sort to scan thresholds
          std::vector<std::size_t> idx(N);
          std::iota(idx.begin(), idx.end(), 0);
          std::sort(idx.begin(), idx.end(),
            [&](auto a, auto b){ return vals[a] < vals[b]; });

          // try thresholds between distinct vals
          double sumPos = 0, sumNeg = 0;
          for (auto wi : w) {
            // nothing here – we'll recompute errors below
          }

          // lets keep a running error for polarity +1 and -1
          double wPos=0, wNeg=0;
          for (auto i : idx) {
            bool label = (i < Npos);
            double weight = w[i];
            // consider threshold at vals[i]
            // for polarity = +1: predict positive if val < thresh
            // error = sum weights where pred != label
            // We can update wPos, wNeg incrementally, but for clarity:
            double thr = vals[i];
            for (int polarity : {+1, -1}) {
              double err = 0;
              for (std::size_t k = 0; k < N; ++k) {
                bool pred = (polarity * vals[k] < polarity * thr);
                if (pred != (k < Npos)) err += w[k];
              }
              if (err < bestErr) {
                bestErr = err;
                bestW = { feat, static_cast<int>(thr), polarity, 0.0 };
              }
            }
          }
        }

        // 2) compute alpha and add weak
        double err = std::max(bestErr, 1e-10);
        double alpha = 0.5 * std::log((1 - err) / err);
        bestW.alpha = alpha;
        stage.add(std::move(bestW));
        sumAlphas += alpha;

        std::cout << "Round " << (r+1) << " complete, best error: " << std::fixed << std::setprecision(4) << bestErr << std::endl;
        std::cout.flush();

        // 3) update weights
        for (std::size_t i = 0; i < N; ++i) {
          auto& weight = w[i];
          bool label = (i < Npos);
          // re-evaluate weak on sample i
          auto& wkl = stage.weaks_.back();  // last weak
          bool pred = (wkl.polarity *
                       (i < Npos ? wkl.feat(posIs[i],0,0)
                                 : wkl.feat(negIs[i-Npos],0,0))
                      < wkl.polarity * wkl.thresh);
          weight *= std::exp(-alpha * (label ? +1 : -1) * (pred ? +1 : -1));
        }
        // normalize
        double Z = std::accumulate(w.begin(), w.end(), 0.0);
        for (auto& weight : w) weight /= Z;
      }

      // 4) set the strong threshold to half the total alpha
      stage.setThreshold(0.5 * sumAlphas);

      cascade.addStage(stage);

      // recompute threshold = 0.5 * sum of alphas
      sumAlphas = 0;
      for (auto const& w : stage.weaks_)
        sumAlphas += w.alpha;
      stage.setThreshold(0.5 * sumAlphas);

      std::cout << "Added stage " << currentStage << " with " << stage.weaks_.size() << " weak classifiers" << std::endl;
      std::cout.flush();

      // evaluate on negatives to filter out "easy" ones
      std::cout << "Evaluating negatives to filter out easy ones..." << std::endl;
      std::cout.flush();

      std::vector<Image<long long>> hardNegs;
      int count = 0;
      for (auto const& I : negIs) {
        if (cascade.classify(I, 0, 0))
          hardNegs.push_back(I);

        // show progress periodically when evaluating negatives
        if (++count % 10 == 0) {
          std::cout << "Evaluated " << count << "/" << negIs.size() << " negatives" << std::endl;
          std::cout.flush();
        }
      }
      negIs.swap(hardNegs);

      // update overall FPR/TPR if desired...
      overallFPR = double(negIs.size()) / std::max<size_t>(1, initial_neg_count);
      std::cout << "Stage " << currentStage << " complete. Hard negatives: " << negIs.size()
                << ", Overall FPR: " << std::fixed << std::setprecision(4) << overallFPR << std::endl;
      std::cout.flush();

      // break if no more negatives
      if (negIs.empty()) {
        std::cout << "No more negative samples, stopping cascade training" << std::endl;
        std::cout.flush();
        break;
      }
    }

    return cascade;
}


// overloadigg  with default progress callback
CascadeClassifier<int>
Trainer::trainCascade(
    std::vector<Image<long long>> posIs,
    std::vector<Image<long long>> negIs,
    const TrainerOptions& opts)
{
    // default empty progress callback
    ProgressCallback noCallback = [](int, int, int, int){};
    return trainCascade(std::move(posIs), std::move(negIs), opts, noCallback);
}

} // namespace vj
