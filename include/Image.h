#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>


//  holds pixel data in a 2D vector
template<typename T>
class Image {
private:
    size_t width_, height_;
    std::vector<std::vector<T>> data_; // each element is a row
public:
    // we initialize constructor explicitly
    Image(size_t width, size_t height) : width_(width), height_(height) {
        data_.resize(height_); // num of rows
        for (auto& row : data_) { // every element is a vector, so we use auto&
            row.resize(width_); // num of columns (row width)
        }
    }

    // getters
    size_t width() const { return width_; }
    size_t height() const { return height_; }

    // explicit access operator (.at bound checking and readibility)
    std::vector<T>& operator[](size_t idx) { return data_.at(idx); }
    const std::vector<T>& operator[](size_t idx) const { return data_.at(idx); }

    // below the integral image computation
    Image<int> computeIntegralImage() const {
        Image<int> integral(width_, height_);
        for (size_t row = 0; row < height_; ++row) {
            int rowSum = 0;
            for (size_t x = 0; x < width_; ++x) {
                rowSum += (*this)[row][x];
                if (row == 0)
                    integral[row][x] = rowSum;
                else
                    integral[row][x] = integral[row - 1][x] + rowSum;
            }
        }
        return integral;
    }
};
