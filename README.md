# Viola-Jones algorithm

Viola-Jones face detection algorithm implementation written on C++

example on how to build:
```
cmake -S . -B build \
  -DENABLE_CLANG_TIDY=ON
cmake --build build
```

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

## Training

First of all, build the project with CMake, then you can train the AdaBoost classifier like this:

```
./build/trainer "train/face/*.pgm" "train/non-face/*.pgm" *name*.dat
```

and then you can use the cascade to detect faces in images or videos:

```
./build/main *name*.dat
```

Cascade file must be inside `build`!
