# Viola-Jones algorithm

Viola-Jones face detection algorithm implementation written on C++

Reusable header-only lib

Project structure:

Project/
├── CMakeLists.txt
├── include/
│   ├── viola_jones/
│   │   ├── Image.h
│   │   ├── HaarFeature.h
│   │   ├── AdaBoost.h
│   │   ├── CascadeClassifier.h
│   │   └── Trainer.h
│   └── utils.hpp                   # loadIntegralSamples, etc.
└── src/
      ├── Trainer.cpp
      ├── trainer_main.cpp            # builds cascade from pos/neg folders
      └── detect_main.cpp             # camera + sliding/multi-scale loop
