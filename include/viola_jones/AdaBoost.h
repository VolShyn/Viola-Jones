#ifndef ADABOOST_HPP
#define ADABOOST_HPP

#include "HaarFeature.h"
#include <vector>
#include <cstddef>

namespace vj {

template<typename T>
class AdaBoost {
public:
    struct Weak {
        HaarFeature<T> feat;
        T           thresh;
        int         polarity;  // ±1
        double      alpha;     // weight
    };

    // add one weak classifier
    void add(Weak w) { weaks_.push_back(std::move(w)); }

    // weighted vote
    bool classify(const Image<long long>& I,
                  std::size_t ox, std::size_t oy) const
    {
        double sum = 0;
        for (auto const& w : weaks_) {
            auto val = w.feat(I, ox, oy);
            int vote = (w.polarity * val < w.polarity * w.thresh) ? 1 : 0;
            sum += vote * w.alpha;
        }
        return sum > threshold_;
    }

    // adjust final threshold if needed
    void setThreshold(double t) { threshold_ = t; }

private:
    std::vector<Weak> weaks_;
    double threshold_ = 0.5;
};

} // namespace vj
#endif // ADABOOST_HPP
