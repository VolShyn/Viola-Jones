#ifndef CASCADE_HPP
#define CASCADE_HPP

#include "AdaBoost.h"
#include "Image.h"
#include <vector>
#include <cstddef>


// stages of the classification into cascade

namespace vj {

template<typename T>
class Cascade {
public:
    void addStage(const AdaBoost<T>& stage) {
        stages_.push_back(stage);
    }

    bool classify(const Image<long long>& I,
                  std::size_t ox, std::size_t oy) const
    {
        for (auto const& st : stages_) {
            if (!st.classify(I, ox, oy))
                return false;
        }
        return true;
    }

private:
    std::vector<AdaBoost<T>> stages_;
};

} // namespace vj

#endif // CASCADE_HPP
