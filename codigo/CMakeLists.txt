cmake_minimum_required(VERSION 3.9.4)
project(tp1)

set(CMAKE_CXX_FLAGS -g)
set(CMAKE_CXX_FLAGS -O2)
set(CMAKE_CXX_STANDARD 14)


# Leemos todos los archivos fuentes en ./src
file(GLOB SOURCE_FILES src/ src/*.h src_catedra/*.h)

# Leemos todos los archivos de test en ./tests
file(GLOB TEST_SOURCES tests/*.cpp)

add_subdirectory(tests/google-test)

# Creamos el ejecutable para correr los tests
add_executable(correrTests ${TEST_SOURCES} ${SOURCE_FILES})

add_executable(tp2 src/main.cpp src/KNNClassifier.h src/PCA.h src/SentimentPredictor.h)

if (APPLE)
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)
    add_definitions(-D__GLIBCXX__)
endif (APPLE)

# Necesitamos asociar los archivos del framework de testing
target_link_libraries(correrTests gtest gtest_main)

add_test(correrTests correrTests)

# Target para correr Valgrind
add_custom_target(correrValgrind
        valgrind --leak-check=full -v ./correrTests 2>&1
        DEPENDS correrTests)
