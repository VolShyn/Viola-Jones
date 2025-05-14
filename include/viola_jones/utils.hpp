#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "viola_jones/Image.h"

// we load all the images that match `glob_pattern` (e.g. "train/face/*.png") and then compute
// their integral images and return as vj::Image<long long>
// target_size is the window size for Haar features (default 24x24)
inline std::vector< vj::Image<long long> >
loadIntegralSamples(const std::string& glob_pattern, int target_size = 24)
{
    std::vector< vj::Image<long long> > samples;
    std::vector<cv::String> files;
    cv::glob(glob_pattern, files);

    std::cout << "Loading " << files.size() << " images from " << glob_pattern << std::endl;

    for (auto const& file : files) {
        cv::Mat img = cv::imread(file, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Warning: Could not load file " << file << std::endl;
            continue;
        }

        // resize image to target_size x target_size
        // actually not needed because the dataset is already resized
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(target_size, target_size));

        // compute integral image (size will be (H+1)x(W+1))
        cv::Mat integral;
        cv::integral(resized, integral, CV_64F);

        // wrap into vjImage resized img size
        int rows = resized.rows, cols = resized.cols;
        vj::Image<long long> I(cols, rows);
        for (int y = 0; y < rows; ++y) {
          for (int x = 0; x < cols; ++x) {
            // integral at (y+1,x+1) holds sum over [0..y][0..x]
            I[y][x] = static_cast<long long>(integral.at<double>(y+1, x+1));
          }
        }
        samples.push_back(std::move(I));
    }

    std::cout << "succesfully loaded " << samples.size() << " samples" << std::endl;
    return samples;
}
