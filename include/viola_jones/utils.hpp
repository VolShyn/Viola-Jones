#pragma once

// needed to prepare sample set

#include <iostream>
#include <opencv2/opencv.hpp>
#include "viola_jones/Image.h"

inline std::vector<vj::Image<long long>>
loadIntegralSamples(const std::string& globPattern) {
    std::vector<cv::String> files;
    cv::glob(globPattern, files);
    std::vector< vj::Image<long long> > out;
    for (auto &f : files) {
        cv::Mat gray = cv::imread(f, cv::IMREAD_GRAYSCALE);
        if (gray.empty() || gray.cols!=24 || gray.rows!=24) continue;
        // wrap into vj::Image<int> then integral → Image<long long>
        vj::Image<int> img(24,24);
        for(int y=0;y<24;++y)
          for(int x=0;x<24;++x)
            img[y][x] = gray.at<unsigned char>(y,x);
        out.push_back( img.integral() );
    }
    return out;
}
