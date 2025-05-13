#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "viola_jones/Image.h"

// we load all the images that match `glob_pattern` (e.g. "train/face/*.png") and then compute
// their integral images and return as vj::Image<long long>
inline std::vector< vj::Image<long long> >
loadIntegralSamples(const std::string& glob_pattern)
{
    std::vector< vj::Image<long long> > samples;
    std::vector<cv::String> files;
    cv::glob(glob_pattern, files);

    for (auto const& file : files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        // compute integral image (size will be (H+1)x(W+1))
        cv::Mat integral;
        cv::integral(img, integral, CV_64F);

        // wrap into vj::Image<long long> of original img size
        int rows = img.rows, cols = img.cols;
        vj::Image<long long> I(cols, rows);
        for (int y = 0; y < rows; ++y) {
          for (int x = 0; x < cols; ++x) {
            // integral at (y+1,x+1) holds sum over [0..y][0..x]
            I[y][x] = static_cast<long long>(integral.at<double>(y+1, x+1));
          }
        }
        samples.push_back(std::move(I));
    }
    return samples;
}
