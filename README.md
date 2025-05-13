# Viola-Jones algorithm

Viola-Jones face detection algorithm implementation written on C++

You can check if all the dependencies are correctly installed compiling the demo

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
  - `main.cpp` — _camera + sliding/multi-scale loop_
  - `trainer_main.cpp` — _builds cascade from *pos/neg* folders_
