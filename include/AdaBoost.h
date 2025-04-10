#include "HaarFeature.h"

#pragma once
template<typename T>
class AdaBoost {
public:
    // a weak classifier
    struct WeakClassifier {
        HaarFeature<T> feature;
        T threshold;
        int polarity; // typically +1 or -1.
        double weight;
    };
};
