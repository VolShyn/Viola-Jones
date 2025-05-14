#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "viola_jones/CascadeClassifier.h"
#include "viola_jones/utils.hpp"

int main(int argc, char** argv){
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <cascade_file>\n";
        return 1;
    }

    // load the trained cascade classifier
    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "erorik: could not open cascade file " << argv[1] << "\n";
        return 1;
    }
    vj::CascadeClassifier<int> cascade = vj::CascadeClassifier<int>::load(in);
    std::cout << "loading cascade classifier from " << argv[1] << "\n";

    // initialize camera (0 = default)
    cv::VideoCapture cap(0);
    if(!cap.isOpened()){
        std::cerr << "Eerroooor, please buy a camera\n";
        return 1;
    }

    // set lower resolution for better performance
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 384);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 288);

    cv::Mat frame, gray, small_frame;
    bool process_this_frame = true;
    int frame_count = 0;

    const int base_window_size = 24; // cascade was trained on this size
    const double scale_factor = 1.3;
    const int min_neighbors = 2;
    const int max_scales = 10;
    const int step_ratio = 4; // increased step size
    const double min_scale = 1.0; // i.e. dont lower the resolution
    const double max_scale = 10.0; // maximum scale
    const double min_face_ratio = 0.05;
    const double max_face_ratio = 0.8;

    for(;;){
        if(!cap.read(frame)){
            std::cerr << "warning: failed to grab frame\n";
            break;
        }

        frame_count++;

        // only process every other frame for better UI responsiveness
        process_this_frame = (frame_count % 2 == 0);

        // display previous detection results when skipping processing
        std::vector<cv::Rect> faces;

        if (process_this_frame) {
            // convert to gray
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            // additional resizing if needed
            const double resize_factor = 1;
            cv::resize(gray, small_frame, cv::Size(), resize_factor, resize_factor);

            const int W = small_frame.cols, H = small_frame.rows;

            // pre-calculate min/max face sizes
            int min_face_size = static_cast<int>(H * min_face_ratio);
            int max_face_size = static_cast<int>(H * max_face_ratio);

            // store detections at each scale
            std::vector<cv::Rect> raw_detections;
            double scale = min_scale;

            // limit total number of scales to process
            int scales_processed = 0;

            while (scales_processed < max_scales && scale <= max_scale) {
                // current detection size at this scale
                int current_size = cvRound(base_window_size * scale);

                // skip if face would be outside our target size range
                if (current_size < min_face_size || current_size > max_face_size) {
                    scale *= scale_factor;
                    continue;
                }

                scales_processed++;

                // create a scaled image for this level in the pyramid
                cv::Mat scaled_img;
                double scale_ratio = base_window_size / static_cast<double>(current_size);
                cv::resize(small_frame, scaled_img, cv::Size(), scale_ratio, scale_ratio);

                // we create vj::Image for this scale
                int scaled_W = scaled_img.cols;
                int scaled_H = scaled_img.rows;
                vj::Image<int> img(scaled_W, scaled_H);

                // copy scaled image data
                for(int y = 0; y < scaled_H; ++y) {
                    const uchar* row_ptr = scaled_img.ptr<uchar>(y);
                    for(int x = 0; x < scaled_W; ++x) {
                        img[y][x] = row_ptr[x];
                    }
                }

                auto I = img.integral();
                int step = std::max(2, base_window_size / step_ratio);

                // scan with sliding window
                for (int y = 0; y + base_window_size <= scaled_H; y += step) {
                    for (int x = 0; x + base_window_size <= scaled_W; x += step) {
                        if (cascade.classify(I, x, y)) {
                            // then we convert back to original image coordinates (account for resize_factor)
                            int orig_x = cvRound(x / resize_factor);
                            int orig_y = cvRound(y / resize_factor);
                            int orig_size = cvRound(current_size / resize_factor);
                            raw_detections.push_back(cv::Rect(orig_x, orig_y, orig_size, orig_size));
                        }
                    }
                }

                // move to next scale
                scale *= scale_factor;
            }

            // we perform non-maximum suppression to merge overlapping detections
            if (!raw_detections.empty()) {
                // we use OpenCV's built-in grouping fщлunction
                cv::groupRectangles(raw_detections, min_neighbors, 0.2);
                faces = raw_detections;
            }
        }

        // draw faces and count
        for (const auto& face_rect : faces) {
            cv::rectangle(frame, face_rect, cv::Scalar(0, 255, 0), 2);
        }

        static int frame_count = 0;
        static double fps = 0;
        static std::chrono::time_point<std::chrono::high_resolution_clock> prev_time;

        if (frame_count == 0) {
            prev_time = std::chrono::high_resolution_clock::now();
        }

        frame_count++;

        if (frame_count >= 10) {
            auto curr_time = std::chrono::high_resolution_clock::now();
            double elapsed_seconds = std::chrono::duration<double>(curr_time - prev_time).count();
            fps = frame_count / elapsed_seconds;

            frame_count = 0;
        }

        cv::putText(frame, "faces: " + std::to_string(faces.size()) + " FPS: " + std::to_string(int(fps)),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Viola-Jones", frame);
        if(cv::waitKey(1) == 27){
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
