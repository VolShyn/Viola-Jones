// training binary file i.e. we need it only to once train our classifier
// and serialize the resulting model. Drawing parallels with Python nn training, it's some kind of a dataloader

#include <iostream>
#include <fstream>
#include <iomanip>
#include "viola_jones/Trainer.h"
#include "viola_jones/utils.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
      std::cerr << "Usage: " << argv[0]
                << " <pos_glob> <neg_glob> <out_cascade_file>\n";
      return 1;
    }


    vj::TrainerOptions opts;
    opts.window_size = 24;
    opts.num_rounds  = 20;

    auto posIs = loadIntegralSamples(argv[1], opts.window_size);
    auto negIs = loadIntegralSamples(argv[2], opts.window_size);
    if (posIs.empty() || negIs.empty()) {
      std::cerr << "Oops no samples loaded\n";
      return 1;
    }

    std::cout << "Training with " << posIs.size() << " positive and "
              << negIs.size() << " negative samples" << std::endl;

    // Define a callback function to track training progress
    auto progressCallback = [](int currentStage, int totalStages, int currentRound, int totalRounds) {
        // calc overall progress (stages contribute 80%, current round in stage contributes 20%)
        float stageProgress = static_cast<float>(currentStage) / std::max(1, totalStages);
        float roundProgress = static_cast<float>(currentRound) / std::max(1, totalRounds);
        float progress = 0.8f * stageProgress + 0.2f * roundProgress;

        // progress bar
        int barWidth = 50;
        std::string bar = "[";
        int pos = static_cast<int>(barWidth * progress);
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) bar += "=";
            else if (i == pos) bar += ">";
            else bar += " ";
        }
        bar += "]";

        // clear the entire line before printing the progress
        std::cout << "\r" << std::string(100, ' ') << "\r";
        std::cout << bar << " " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% "
                  << "Stage " << currentStage << "/" << totalStages
                  << " Round " << currentRound << "/" << totalRounds;
        std::cout.flush();
    };

    // start the training with our progress callback
    std::cout << "Starting cascade training (this may take a while)..." << std::endl;
    std::cout.flush();

    auto cascade = vj::Trainer::trainCascade(posIs, negIs, opts, progressCallback);

    // complete the progress bar at 100%
    int barWidth = 50;
    std::cout << "\n[" << std::string(barWidth, '=') << "] 100.0% Complete!       " << std::endl;

    std::ofstream out(argv[3]);
    cascade.save(out);
    std::cout << "cascade written to " << argv[3] << "\n";
    return 0;
}
