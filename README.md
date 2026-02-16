# 📚 ArUco Nano

A minimalist, header-only, high-performance C++ library for ArUco marker detection.

  

## 🏛 Introduction

ArUco Nano is a lightweight (single header, <500 lines) implementation of the ArUco marker detection algorithm. Designed for speed and ease of integration, it replaces general-purpose OpenCV algorithms with a specialized Visited-Aware Contour Tracing algorithm and direct sub-pixel code sampling.

Key advantages over the standard OpenCV implementation:


* 🚀 High Performance: Up to  6.5x faster than standard OpenCV ArUco (single-threaded) and 2x faster than the multi-threaded implementation.
* 🪶 Header-Only: No complex build systems or linking required. Just #include "aruco_nano.h".
* 🔌Drop-in Replacement: Includes an ArucoDetector wrapper that mimics the standard OpenCV API for easy migration.
* 🔬Robust: Higher F1 score on challenging datasets.

## Installation

Integration is trivial. Simply copy the aruco_nano.h file into your project directory.

### Dependencies
* OpenCV (Core, ImgProc, Calib3d, HighGUI)
* C++17 standard

## Usage

### 1. Basic Detection (Nano Style)
The simplest way to detect markers using the default dictionary (DICT_ARUCO_MIP_36h12).

```cpp
#include <opencv2/highgui.hpp>
#include "aruco_nano.h"

int main() {
    cv::Mat image = cv::imread("image.jpg");
    
    // Detect markers
    auto markers = aruco_nano::MarkerDetector::detect(image);
    
    // Draw and save
    for(const auto &m : markers) 
        m.draw(image);
        
    cv::imwrite("output.jpg", image);
    return 0;
}
```

### 2. OpenCV Compatibility Mode
If you have existing code using cv::aruco::ArucoDetector, you can switch to ArUco Nano with minimal changes.

```cpp
#include <opencv2/highgui.hpp>
#include "aruco_nano.h"

int main() {
    cv::Mat image = cv::imread("image.jpg");
    
    // Use standard OpenCV dictionary
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    
    // Initialize Nano detector with OpenCV API wrapper
    aruco_nano::ArucoDetector detector(dictionary);
    
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    
    // Detect
    detector.detectMarkers(image, corners, ids);
    
    // Draw using standard OpenCV function
    cv::aruco::drawDetectedMarkers(image, corners, ids);
    return 0;
}
```

### 3. Pose Estimation

```cpp
// Assuming 'markers' are detected
cv::Mat cameraMatrix, distCoeffs; // Load from calibration
float markerSize = 0.05f; // 5cm

for(const auto &m : markers){
    auto pose = m.estimatePose(cameraMatrix, distCoeffs, markerSize);
    cv::Mat rvec = pose.first;
    cv::Mat tvec = pose.second;
    
    std::cout << "Rvec: " << rvec << " Tvec: " << tvec << std::endl;
}
```

## 🚀 Performance

ArUco Nano significantly outperforms the standard OpenCV implementation, particularly at high resolutions where memory bandwidth and contour extraction become bottlenecks.

Speedup vs OpenCV (Single-Threaded):

| Resolution | ArUco Nano Time | OpenCV Time | Speedup |
|:----------:|:---------------:|:-----------:|:-------:|
| 1 MP | 6.68 ms         | 43.46 ms    | 6.5x|
| 4 MP | 22.93 ms        | 125.00 ms   | 5.4x|
| 16 MP | 101.67 ms       | 504.27 ms   | 4.9x|

Benchmarks performed on an Intel Core i7-13700H.

## 🧠 Reproducing Benchmarks

This repository includes a performance testing tool (testperf.cpp).

### 🏗 Build
```
bash
mkdir build && cd build
cmake ..
make
```

### 🏃 Run
To strictly reproduce the paper results (preventing thermal throttling), run:

```
bash
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
sudo cpupower frequency-set -u 4000MHz
./testperf <path_to_image_directory> [-show] [-scale 0.5]
```

## 📚 Citation

If you use ArUco Nano in your research, please cite the following paper:

> R. Muñoz-Salinas, F. J. Romero-Ramirez, S. Garrido-Jurado, "ArUco Nano: a simpler, faster, and more reliable fiducial marker detector", TO BE PUBLISHED, 2026.

Additionally, please cite the foundational ArUco papers:

1.  S. Garrido-Jurado et al., "Automatic generation and detection of highly reliable fiducial markers under occlusion", Pattern Recognition, 2014.
2.  F.J. Romero-Ramirez et al., "Speeded up detection of squared fiducial markers", Image and Vision Computing, 2018.

## 🔖 License

This project is licensed under the Apache License 2.0. You can freely use the code in your commercial products.

## Acknowledgments

This research has been funded by the PID2023-147296NB-I00 project of the Ministry of Science, Innovation, and Universities of Spain.
