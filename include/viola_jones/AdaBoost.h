#ifndef ADABOOST_HPP
#define ADABOOST_HPP

#include "HaarFeature.h"
#include <vector>
#include <cstddef>
#include <string>

namespace vj {

template<typename T>
class AdaBoost {
    friend class Trainer; // Allow Trainer to access private members
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

    // serialization
    void save(std::ostream& os) const {
      // number of weak learners
      os << weaks_.size() << "\n";
      for (auto const& w : weaks_) {
        // 1) serialize feature
        w.feat.save(os);
        // 2) serialize thresh, polarity, alpha
        os << w.thresh << " "
           << w.polarity << " "
           << w.alpha << "\n";
      }
      // 3) serialize threshold
      os << threshold_ << "\n";
    }

    static AdaBoost<T> load(std::istream& is) {
      AdaBoost<T> ab;
      std::string K_str;
      is >> K_str;
      size_t K = std::stoull(K_str);
      for (size_t i = 0; i < K; ++i) {
        // a) load feature
        auto feat = HaarFeature<T>::load(is);
        // b) load thresh, polarity, alpha
        T thresh; int polarity; double alpha;
        is >> thresh >> polarity >> alpha;
        ab.weaks_.push_back({feat, thresh, polarity, alpha});
      }
      // load threshold
      is >> ab.threshold_;
      return ab;
    }

private:
    std::vector<Weak> weaks_;
    double threshold_ = 0.5;
};

} // namespace vj
#endif // ADABOOST_HPP
