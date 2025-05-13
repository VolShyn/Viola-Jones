#include <iostream>
#include <opencv2/opencv.hpp>

#include "viola_jones/Image.h"
#include "viola_jones/HaarFeature.h"
#include "viola_jones/AdaBoost.h"
#include "viola_jones/CascadeClassifier.h"

int main(int argc, char** argv){
    // just comment that we need path
    if(argc < 2){
        std::cerr << "usage: " << argv[0] << " <image_path>\n";
        return 1;
    }

    // loading image in grayscale
    cv::Mat mat = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if(mat.empty()){
        std::cerr << "error: could not load image " << argv[1] << "\n";
        return 1;
    }

    cv::imshow("the image", mat);
    int k = cv::waitKey(0); // wait

    // wrapping cv::Mat into vj::Image<int>
    // so we can reuse the same integral/Haar code
    const int W = mat.cols;
    const int H = mat.rows;
    vj::Image<int> img(W, H);
    for(int y = 0; y < H; ++y){
        for(int x = 0; x < W; ++x){
            img[y][x] = mat.at<unsigned char>(y, x);
        }
    }

    // integral image
    auto I = img.integral();

    // define a trivial Haar feature and classifier stage
    // explicit casts to size_t
    vj::Rect white {
        0u,
        0u,
        static_cast<std::size_t>( std::min(W, 24) ),
        static_cast<std::size_t>( std::min(H, 24) / 2 )
    };

    vj::Rect black {
        0u,
        static_cast<std::size_t>( std::min(H, 24) / 2 ),
        static_cast<std::size_t>( std::min(W, 24) ),
        static_cast<std::size_t>( std::min(H, 24) / 2 )
    };

    vj::HaarFeature<int> hf{white, black};

    vj::AdaBoost<int> stage;
    stage.add({ hf,  1000, +1, 0.6 });        // dummy parameters
    stage.setThreshold(0.5);

    vj::Cascade<int> cascade;
    cascade.addStage(stage);

    // slide a single 24×24 window at (0,0) for demo
    bool face = false;
    if(W >= 24 && H >= 24){
      face = cascade.classify(I, 0, 0);
    }
    std::cout << "detect at (0,0): " << (face? "face":"not face") << "\n";

    return 0;
}
