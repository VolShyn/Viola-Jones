#include <iostream>
#include <opencv2/opencv.hpp>

#include "viola_jones/Image.h"
#include "viola_jones/HaarFeature.h"
#include "viola_jones/AdaBoost.h"
#include "viola_jones/CascadeClassifier.h"

int main(int argc, char** argv){
    // initialize camera (0 = default)
    cv::VideoCapture cap(0);
    if(!cap.isOpened()){
        std::cerr << "error: could not open camera\n";
        return 1;
    }

    // prepare your weak classifier cascade once
    vj::Rect white { 0u, 0u, 24u, 12u };
    vj::Rect black { 0u, 12u, 24u, 12u };
    vj::HaarFeature<int> hf{white, black};

    vj::AdaBoost<int> stage;
    stage.add({ hf, 1000, +1, 0.6 });
    stage.setThreshold(0.5);

    vj::Cascade<int> cascade;
    cascade.addStage(stage);

    cv::Mat frame, gray;
    for(;;){
        if(!cap.read(frame)){
            std::cerr << "warning: failed to grab frame\n";
            break;
        }

        // convert to gray
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        const int W = gray.cols, H = gray.rows;

        // wrap into vj::Image
        vj::Image<int> img(W, H);
        for(int y = 0; y < H; ++y){
            for(int x = 0; x < W; ++x){
                img[y][x] = gray.at<unsigned char>(y, x);
            }
        }

        // build integral image
        auto I = img.integral();

        // classify at (0,0) if frame large enough
        bool face = false;
        if(W >= 24 && H >= 24){
            face = cascade.classify(I, 0, 0);
        }

        // draw feedback
        if(face){
            // draw a green 24×24 box at the top-left
            cv::rectangle(frame, cv::Point(0,0), cv::Point(24,24),
                          cv::Scalar(0,255,0), 2);
            cv::putText(frame, "face", {30,20},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0,255,0), 1);
        } else {
            cv::putText(frame, "Not face", {5, H-10},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0,0,255), 1);
        }

        // show result
        cv::imshow("Viola-Jones", frame);
        if(cv::waitKey(30) == 27){ // ESC to quit
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
