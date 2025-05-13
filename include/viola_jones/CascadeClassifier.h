#ifndef CASCADE_HPP
#define CASCADE_HPP

#include "AdaBoost.h"
#include "Image.h"
#include <vector>
#include <cstddef>
#include <string>


// stages of the classification into cascade

namespace vj {

template<typename T>
class CascadeClassifier {
public:
    //  one stage “strong classifier”
    void addStage(const AdaBoost<T>& stage) {
        stages_.push_back(stage);
    }

    // the decision threshold (half the sum of alphas)
    void setThreshold(double t) {
        threshold_ = t;
    }

    // the cascade on an integral‐image window at (x,y)
    bool classify(const Image<long long>& I, std::size_t x, std::size_t y) const {
        double sum = 0;
        for (auto const& stage : stages_) {
            if (!stage.classify(I, x, y))
                return false;  // early reject
        }
        return true;
    }

    // offline serialization
    void save(std::ostream& os) const {
        os << stages_.size() << "\n";
        for (auto const& s : stages_)
            s.save(os);
        os << threshold_ << "\n";
    }


    // same but deserialization
    static CascadeClassifier<T> load(std::istream& is) {
        CascadeClassifier<T> c;
        std::string S_str;
        is >> S_str;
        size_t S = std::stoull(S_str);
        for (size_t i = 0; i < S; ++i)
            c.stages_.push_back(AdaBoost<T>::load(is));
        is >> c.threshold_;
        return c;
    }

private:
    std::vector<AdaBoost<T>> stages_;
    double threshold_{0.0};
};

} // namespace vj

#endif // CASCADE_HPP
