#include <iostream>
#include <vector>
#include <stdexcept>
#pragma once

//  holds pixel data in a 2D vector
template<typename T>
class Image {
public:
    using value_type = T;

    Image(size_t width, size_t height) : width_(width), height_(height) {
        data_.resize(height_);
        for (auto& row : data_) {
            row.resize(width_);
        }
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    // access operator
    std::vector<T>& operator[](size_t idx) { return data_.at(idx); }
    const std::vector<T>& operator[](size_t idx) const { return data_.at(idx); }

    // below the integral image computation
    // we use a larger type (long long) for robustness
    Image<long long> computeIntegralImage() const {
        Image<long long> integral(width_, height_);
        for (size_t y = 0; y < height_; ++y) {
            long long rowSum = 0;
            for (size_t x = 0; x < width_; ++x) {
                rowSum += (*this)[y][x];
                if (y == 0)
                    integral[y][x] = rowSum;
                else
                    integral[y][x] = integral[y - 1][x] + rowSum;
            }
        }
        return integral;
    }

private:
    size_t width_, height_;
    std::vector<std::vector<T>> data_;
};
