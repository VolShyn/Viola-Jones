#include "../include/Image.h"

#include <iostream>
#include <ostream>

int main() {

    const size_t rows = 6;
    const size_t cols = 6;

    Image<int> img(cols, rows);

    const char* rowValues[rows] = {
        "314159",
        "265358",
        "979323",
        "846264",
        "338327",
        "950288"
    };


    // for each row and column, convert the digit from the string to int
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // sub 0 converts the character digit to its corresponding int value
            img[y][x] = rowValues[y][x] - '0';
        }
    }

    auto integral = img.computeIntegralImage();

    std::cout << "Integral image:" << std::endl;
    for (int y = 0; y < integral.height(); ++y) {
        for (int x = 0; x < integral.width(); ++x) {
            std::cout << integral[y][x] << "\t";
        }
        std::cout << "\n";
    }

    return 0;
};
