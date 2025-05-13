# Viola-Jones algorithm

Viola-Jones face detection algorithm implementation written on C++

Reusable header-only lib

Project structure:

**Project/**
- **CMakeLists.txt**
- **include/**
  - **viola_jones/**
    - `Image.h`
    - `HaarFeature.h`
    - `AdaBoost.h`
    - `CascadeClassifier.h`
    - `Trainer.h`
  - `utils.hpp` — helper utilities (e.g., `loadIntegralSamples`)
- **src/**
  - `Trainer.cpp`
  - `trainer_main.cpp` — _builds cascade from *pos/neg* folders_
  - `detect_main.cpp` — _camera + sliding/multi-scale loop_
