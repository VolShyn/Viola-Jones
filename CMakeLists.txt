cmake_minimum_required(VERSION 3.15)
project(ViolaJonesLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find pOpenCV
find_package(OpenCV REQUIRED)

# clang-tidy
option(ENABLE_CLANG_TIDY "Run clang-tidy" ON)
if(ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy)
    if(CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
    endif()
endif()

# sanitizers
option(ENABLE_SANITIZERS "Enable ASan/UBSan" ON)
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
    -fsanitize=address
    -fsanitize=undefined
    -fno-omit-frame-pointer
  )
    add_link_options(
    -fsanitize=address
    -fsanitize=undefined
  )
endif()

# lib target
add_library(viola_jones INTERFACE)
target_include_directories(viola_jones
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# trainer exec
add_executable(trainer
  src/trainer_main.cpp
  src/Trainer.cpp
)
target_link_libraries(trainer
  PRIVATE viola_jones ${OpenCV_LIBS}
)
target_include_directories(trainer
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)

# main
add_executable(main
    src/main.cpp
)
target_link_libraries(main PRIVATE viola_jones ${OpenCV_LIBS})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
