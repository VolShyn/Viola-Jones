#include "../include/Image.h"


int main() {
    // a dummy 5x5 grayscale img
    Image<int> img(5, 5);
    for (size_t y = 0; y < img.height(); ++y) {
        for (size_t x = 0; x < img.width(); ++x) {
            img[y][x] = static_cast<int>(x + y); // foo pixel value
        }
    }

    // integrated image (summed-area table)
    auto integral = img.computeIntegralImage();

    return 0;
};
