#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <vector>
#include <cstddef>

namespace vj {

template<typename T>
class Image {
public:
    using value_type = T;

    Image(std::size_t w, std::size_t h)
      : width_(w), height_(h), data_(h, std::vector<T>(w)) {}

    std::size_t width() const  { return width_; }
    std::size_t height() const { return height_; }

    std::vector<T>&       operator[](std::size_t y)       { return data_[y]; }
    const std::vector<T>& operator[](std::size_t y) const { return data_[y]; }

    // building summed‐area table (what they call an integral image)
    Image<long long> integral() const {
        Image<long long> I(width_, height_);
        for (std::size_t y = 0; y < height_; ++y) {
            long long row_sum = 0;
            for (std::size_t x = 0; x < width_; ++x) {
                row_sum += (*this)[y][x];
                I[y][x] = row_sum + (y>0 ? I[y-1][x] : 0);
            }
        }
        return I;
    }

private:
    std::size_t width_, height_;
    std::vector<std::vector<T>> data_;
};

} // namespace vj

#endif // IMAGE_HPP
