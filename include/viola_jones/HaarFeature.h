#ifndef HAAR_HPP
#define HAAR_HPP

#include "Image.h"
#include <stdexcept>
#include <cstddef>

namespace vj {

struct Rect {
    std::size_t x, y, w, h;
};

template<typename T>
class HaarFeature {
public:
    HaarFeature(Rect white, Rect black)
      : white_(white), black_(black) {}

    // evaluate at offset (ox,oy) on an integral image
    long long operator()(const Image<long long>& I,
                         std::size_t ox, std::size_t oy) const
    {
        return rectSum(I, white_, ox,oy)
             - rectSum(I, black_, ox,oy);
    }

private:
    Rect white_, black_;

    static long long rectSum(const Image<long long>& I,
                             Rect r, std::size_t ox, std::size_t oy)
    {
        std::size_t x1 = ox + r.x, y1 = oy + r.y;
        std::size_t x2 = x1 + r.w - 1, y2 = y1 + r.h - 1;
        if (x2 >= I.width() || y2 >= I.height())
            throw std::out_of_range("HaarFeature out of bounds");

        auto A = (x1>0 && y1>0) ? I[y1-1][x1-1] : 0LL;
        auto B = (y1>0) ? I[y1-1][x2] : 0LL;
        auto C = (x1>0) ? I[y2][x1-1] : 0LL;
        auto D = I[y2][x2];

        return D + A - B - C;
    }
};

} // namespace vj

#endif // HAAR_HPP
