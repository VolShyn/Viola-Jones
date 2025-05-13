#include "viola_jones/Trainer.h"
#include <algorithm>
#include <limits>
#include <cmath>

namespace vj {

// here we generate every two-rectangle Haar feature in a window_size×window_size region
static
std::vector<HaarFeature<int>>
makeAllHaarFeatures(std::size_t window_size)
{
    std::vector<HaarFeature<int>> feats;
    // for simplicity: only horizontal two-rectangles
    for (std::size_t w = 1; w <= window_size; ++w) {
      for (std::size_t h = 1; h*2 <= window_size; ++h) {
        for (std::size_t x = 0; x + w <= window_size; ++x) {
          for (std::size_t y = 0; y + 2*h <= window_size; ++y) {
            Rect white{ x,   y, w, h     };
            Rect black{ x, y+h, w, h     };
            feats.emplace_back(white, black);
          }
        }
      }
    }
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
              bestW = { feat, static_cast<T>(thr), polarity, 0.0 };
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
                     (i < Npos ? feat(posIs[i],0,0)
                               : feat(negIs[i-Npos],0,0))
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
Cascade<int>
Trainer::trainCascade(
    std::vector<Image<long long>> posIs,
    std::vector<Image<long long>> negIs,
    const TrainerOptions& opts)
{
    Cascade<int> cascade;
    double overallFPR = 1.0, overallTPR = 1.0;

    while (overallFPR > 0.01) {  // stop when cascade is good enough
      auto stage = trainStage(posIs, negIs, opts);
      cascade.addStage(stage);

      // evaluate on negatives to filter out “easy” ones
      std::vector<Image<long long>> hardNegs;
      for (auto const& I : negIs) {
        if (cascade.classify(I, 0, 0))
          hardNegs.push_back(I);
      }
      negIs.swap(hardNegs);

      // update overall FPR/TPR if desired...
      overallFPR = double(negIs.size()) / /*initial neg count*/ 1.0;
      // break if no more negatives
      if (negIs.empty()) break;
    }

    return cascade;
}

} // namespace vj
