#ifndef HAAR_HPP
#define HAAR_HPP

#include "Image.h"
#include <stdexcept>
#include <cstddef>

namespace vj {

template<typename T>
struct Rect { T x,y,w,h; };


template<typename T>
class HaarFeature {
public:
    HaarFeature(Rect<T> white, Rect<T> black)
      : white_(white), black_(black) {}

    // evaluate at offset (ox,oy) on an integral image
    long long operator()(const Image<long long>& I,
                         std::size_t ox, std::size_t oy) const
    {
        return rectSum(I, white_, ox,oy)
             - rectSum(I, black_, ox,oy);
    }

    // serialization
    void save(std::ostream& os) const {
      // write white rect then black rect
      os << white_.x << " " << white_.y << " "
         << white_.w << " " << white_.h << "  ";
      os << black_.x << " " << black_.y << " "
         << black_.w << " " << black_.h << "\n";
    }

    static HaarFeature<T> load(std::istream& is) {
      Rect<T> w, b;
      is >> w.x >> w.y >> w.w >> w.h
         >> b.x >> b.y >> b.w >> b.h;
      return HaarFeature<T>(w,b);
    }


private:
    Rect<T> white_, black_;

    static long long rectSum(const Image<long long>& I,
                             Rect<T> r, std::size_t ox, std::size_t oy)
    {
        std::size_t x1 = ox + static_cast<std::size_t>(r.x);
        std::size_t y1 = oy + static_cast<std::size_t>(r.y);
        std::size_t x2 = x1 + static_cast<std::size_t>(r.w) - 1;
        std::size_t y2 = y1 + static_cast<std::size_t>(r.h) - 1;
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
