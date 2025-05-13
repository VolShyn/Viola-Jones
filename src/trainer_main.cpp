// training binary file i.e. we need it only to once train our classifier
// and serialize the resulting model. Drawing parallels with Python nn training, it's some kind of a dataloader

#include <iostream>
#include <fstream>
#include "viola_jones/Trainer.h"
#include "viola_jones/utils.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
      std::cerr << "Usage: " << argv[0]
                << " <pos_glob> <neg_glob> <out_cascade_file>\n";
      return 1;
    }
    auto posIs = loadIntegralSamples(argv[1]);
    auto negIs = loadIntegralSamples(argv[2]);
    if (posIs.empty() || negIs.empty()) {
      std::cerr << "Oops, no samples loaded\n";
      return 1;
    }

    vj::TrainerOptions opts;
    opts.window_size = 24;
    opts.num_rounds  = 20;
    auto cascade = vj::Trainer::trainCascade(posIs, negIs, opts);

    std::ofstream out(argv[3]);
    cascade.save(out);
    std::cout << "cascade written to " << argv[3] << "\n";
    return 0;
}
