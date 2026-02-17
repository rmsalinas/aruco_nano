/**
 * @mainpage ArUco Performance Benchmarking Tool
 * * @section intro_sec Introduction
 * This program evaluates and compares the performance of different ArUco marker detection methods:
 * - Standard OpenCV ArUco (Single-threaded)
 * - Standard OpenCV ArUco (Multi-threaded)
 * - ArUco Nano (Custom implementation)
 * * @section usage_sec Usage
 * Run the executable from the command line:
 * @code{.sh}
 * echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
 * sudo cpupower frequency-set -u 4000MHz
 * ./testperf <path_to_data_directory> [options]
 * @endcode
 * * @subsection opts_sec Options
 * - `-show`: Enable data visualization mode.
 * - `-fresh`: Deletes existing `nanoperf.csv` before starting a new run.
 * - `-scale <val>`: Resizes input images by the specified floating-point factor (e.g., 0.5 for half size).
 * * @section output_sec Output
 * The program generates a CSV file named `nanoperf.csv` containing:
 * - True Positives (TP)
 * - False Positives (FP)
 * - False Negatives (FN)
 * - Execution time in milliseconds (ms)
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <cmath>
#include <map>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "json.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>
#include "aruco_nano.h"

using json = nlohmann::json;

int testNTimesOuter=3;
int testNTimesInner=5;
float scale=1.0;
bool showData=false;
bool singleImage=false;
bool fresh=false;


struct MethodResult{
    std::string name;
    int fp=0, fn=0,tp=0;
    double time_ms=0;
};


struct ImageResult{
    std::string file;
    std::map<std::string,MethodResult> MethodsResults;
};




// Structure to ease comparison
struct MarkerInfo {
    int id;
    std::vector<cv::Point2f> corners;
};
std::vector<MarkerInfo> readFromJsonFile(std::string path) ;

// Helper to calculate the center of a marker from its 4 corners
cv::Point2f getMarkerCenter(const std::vector<cv::Point2f>& corners) {
    cv::Point2f center(0, 0);
    if (corners.empty()) return center;
    for (const auto& p : corners) center += p;
    center *= (1.0 / (float)corners.size());
    return center;
}

/**
 * Computes TP, FP, and FN by comparing detected markers against ground truth.
 * A match requires the same ID and a center distance <= 10 pixels.
 */
void evaluateDetection(const std::vector<int> & detected_ids,const std::vector<std::vector<cv::Point2f>> & detected_corners,
                       const std::vector<MarkerInfo>& groundTruth,
                       int& tp, int& fp, int& fn) {

    std::vector<MarkerInfo> detected;
    for(size_t i=0;i<detected_ids.size();i++){
        detected.push_back({detected_ids[i],detected_corners[i]});
    }
    tp = 0;
    fp = 0;
    std::vector<bool> gtMatched(groundTruth.size(), false);
    for (const auto& det : detected) {
        bool foundMatch = false;
        auto curcenter=getMarkerCenter(det.corners);
        for (size_t j = 0; j < groundTruth.size(); ++j) {
            if (!gtMatched[j] && det.id == groundTruth[j].id) {
                // Calculate GT center (aruco_nano::Marker inherits from std::vector<cv::Point2f>)
                cv::Point2f gtCenter = getMarkerCenter(groundTruth[j].corners);

                double dist = cv::norm(curcenter - gtCenter);
                if (dist <= 10.0) { // 10 pixels error threshold
                    tp++;
                    gtMatched[j] = true;
                    foundMatch = true;
                    break;
                }
            }
        }
        if (!foundMatch) fp++;
    }
    // Any ground truth marker not matched is a False Negative
    fn = (int)groundTruth.size() - tp;
}
int main(int argc, char** argv) {
    // cv::setNumThreads(1);
    // 1. Define the path to your image
    if(argc<2){
        std::cout<<"Usage: "<<argv[0]<<" <path_to_flyinaruco_dir> [-show] [-fresh] [-scale val]"<<std::endl;
        return -1;
    }
    //read optional args
    for(int i=2;i<argc;i++){
        std::string arg=argv[i];
        if(arg=="-show"){
            showData=true;
            std::cout<<"Showing data mode enabled."<<std::endl;
        }
        if(arg=="-fresh"){
            fresh=true;
        }
        if(arg=="-scale" && i+1<argc){
            scale=std::stof(argv[i+1]);
            std::cout<<"Scaling images by "<<scale<<std::endl;
            i++;
        }
    }


    std::string filename="nanoperf_"+std::to_string(scale)+".csv";
    std::ofstream outCSV;
    bool hasHeader=false;

    //delete existing csv file
    if( fresh && std::filesystem::exists(filename)){
        std::filesystem::remove(filename);
        std::cout<<"Fresh mode enabled. Existing csv deleted."<<std::endl;
    }


    if( std::filesystem::exists(filename)){
        hasHeader=true;//assumes it has
        outCSV.open(filename,std::ios::app);
    }
    else{//opens to append
        outCSV.open(filename );

    }


    //read all image names in the file to skip them later
    std::vector<std::string> processedImages;
    {
        std::ifstream inCSV(filename);
        std::string line;
        //skip header
        std::getline(inCSV,line);
        while(std::getline(inCSV,line)){
            std::string imageName=line.substr(0,line.find(","));
            processedImages.push_back(imageName);
        }
    }


    std::vector<std::filesystem::path> images;

    //if dir read all .png images in a folder and iterate
    if( std::filesystem::is_directory(argv[1]) ){
        for(const auto & entry : std::filesystem::directory_iterator(argv[1]))
        {
            std::string path=entry.path().string();
            if(path.find(".jpg")!=std::string::npos){
                //find a file ending in json with the same name
                //if so, add the entry
                if(std::filesystem::exists(path.substr(0,path.size()-4)+".json"))
                    images.push_back(entry.path());
            }
        }
        //sort the images
        std::sort(images.begin(),images.end());
    }
    else if (std::filesystem::is_regular_file(argv[1]) ){
        images.push_back(std::filesystem::path(argv[1]));
        singleImage=true;
    }
    auto nthreads=cv::getNumThreads();//original number of threads

    for(auto image:images){
        //if image already processed, skip
        if(!singleImage && !showData && std::find(processedImages.begin(),processedImages.end(),image.string())!=processedImages.end()){
            std::cout<<"Image "<<image.string()<<" already processed, skipping."<<std::endl;
            continue;
        }

        cv::Mat inputImage = cv::imread(image.string(),cv::IMREAD_GRAYSCALE);


        if( inputImage.empty()){
            std::cout<<"Could not open image: "<<image.string()<<std::endl;
            continue;
        }
        // --- 1. Load Ground Truth ---
        std::string jsonPath = image.string().substr(0, image.string().size() - 4) + ".json";
        std::vector<MarkerInfo> groundTruth = readFromJsonFile(jsonPath); //

        ImageResult ThisImgResult;
        ThisImgResult.file=image.string();

        cv::resize(inputImage,inputImage,cv::Size(float(inputImage.cols)*scale,float(inputImage.rows)*scale));
        // Check if image loaded successfully
        if (inputImage.empty()) {
            std::cout << "Error: Could not open or find the image!" << std::endl;
            return -1;
        }
        std::cout<<"Testing image "<<image.string() <<std::endl;
        for(int nt=0;nt<testNTimesOuter;nt++)  {

            std::string method_name;

            //Opencv 1 Thread
            method_name="01:cv::aruco(1)";
            {
                cv::setNumThreads(1);
                cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
                cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
                detectorParams.errorCorrectionRate=0;
                cv::aruco::ArucoDetector detector(dictionary, detectorParams);
                std::vector<int> markerIds;
                std::vector<std::vector<cv::Point2f>> markerCorners;
                int64 besttime=std::numeric_limits<int64>::max();
                for(int i=0;i<testNTimesInner;i++){
                    auto start=cv::getTickCount();
                    // 4. Perform Detection
                    detector.detectMarkers(inputImage, markerCorners, markerIds);
                    auto end=cv::getTickCount();
                    besttime=std::min(besttime,(end-start));
                }
                ThisImgResult.MethodsResults[method_name].name=method_name;
                ThisImgResult.MethodsResults[method_name].time_ms=double(besttime)*1000.0/cv::getTickFrequency();

                if(nt==testNTimesOuter-1){

                    evaluateDetection( markerIds,  markerCorners,groundTruth,
                                      ThisImgResult.MethodsResults[method_name].tp,
                                      ThisImgResult.MethodsResults[method_name].fp,
                                      ThisImgResult.MethodsResults[method_name].fn);
                }
            }

            //Opencv N Threads
            method_name="02:cv::aruco("+std::to_string(nthreads)+")"    ;
            {
                cv::setNumThreads(nthreads);
                cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
                cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
                detectorParams.errorCorrectionRate=0;
                cv::aruco::ArucoDetector detector(dictionary, detectorParams);
                std::vector<int> markerIds;
                std::vector<std::vector<cv::Point2f>> markerCorners;
                int64 besttime=std::numeric_limits<int64>::max();
                for(int i=0;i<testNTimesInner;i++){
                    auto start=cv::getTickCount();
                    // 4. Perform Detection
                    detector.detectMarkers(inputImage, markerCorners, markerIds);
                    auto end=cv::getTickCount();
                    besttime=std::min(besttime,(end-start));
                }
                ThisImgResult.MethodsResults[method_name].name=method_name;
                ThisImgResult.MethodsResults[method_name].time_ms=double(besttime)*1000.0/cv::getTickFrequency();
                if(nt==testNTimesOuter-1){
                    evaluateDetection( markerIds,  markerCorners,groundTruth,
                                      ThisImgResult.MethodsResults[method_name].tp,
                                      ThisImgResult.MethodsResults[method_name].fp,
                                      ThisImgResult.MethodsResults[method_name].fn);
                }
            }


                //Opencv N Threads
            method_name="03:cv::aruco_nano"   ;
            {
                cv::setNumThreads(1);
                cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
                aruco_nano::ArucoDetector detector(dictionary);
                std::vector<int> markerIds;
                std::vector<std::vector<cv::Point2f>> markerCorners;
                int64 besttime=std::numeric_limits<int64>::max();
                for(int i=0;i<testNTimesInner;i++){
                    auto start=cv::getTickCount();
                    // 4. Perform Detection
                    detector.detectMarkers(inputImage, markerCorners, markerIds);
                    auto end=cv::getTickCount();
                    besttime=std::min(besttime,(end-start));
                }
                ThisImgResult.MethodsResults[method_name].name=method_name;
                ThisImgResult.MethodsResults[method_name].time_ms=double(besttime)*1000.0/cv::getTickFrequency();
                if(nt==testNTimesOuter-1){
                    evaluateDetection( markerIds,  markerCorners,groundTruth,
                                      ThisImgResult.MethodsResults[method_name].tp,
                                      ThisImgResult.MethodsResults[method_name].fp,
                                      ThisImgResult.MethodsResults[method_name].fn);
                }
            }

        }


        //creates the Csv header
        if(!hasHeader){
            hasHeader=true;
            outCSV<<"file,";
            for(auto m:ThisImgResult.MethodsResults){
                outCSV<<m.first<<"_TP,"
                       <<m.first<<"_FP,"
                       <<m.first<<"_FN,"
                       <<m.first<<"_time_ms," ;
            }
            outCSV<<std::endl;
        }
        //now, write the data
        outCSV<<ThisImgResult.file<<",";
        for(auto m:ThisImgResult.MethodsResults){
            outCSV<<m.second.tp<<","
                   <<m.second.fp<<","
                   <<m.second.fn<<","
                   <<m.second.time_ms<<"," ;
        }
        outCSV<<std::endl;
        outCSV.flush();
        //also to cout
        std::cout<<ThisImgResult.file<<std::endl;
        for(auto m:ThisImgResult.MethodsResults){
            std::cout << "[Metrics] "<<m.first<<" : TP=" << m.second.tp
                      << " FP=" << m.second.fp
                      << " FN=" << m.second.fn
                      << " Time="<<m.second.time_ms <<std::endl;
        }
    }
    return 0;
}
std::vector<MarkerInfo> readFromJsonFile(std::string path) {
    std::vector<MarkerInfo> markersList;

    // Abrir el archivo
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo " << path << std::endl;
        return markersList;
    }

    try {
        json data;
        file >> data; // Parsear el contenido del JSON

        // Acceder al array "markers"
        if (data.contains("markers") && data["markers"].is_array()) {
            for (const auto& item : data["markers"]) {
                MarkerInfo marker;
                marker.id = item["id"];
                //                marker.rot = item["rot"];

                // Leer los corners
                for (const auto& corner : item["corners"]) {
                    // corner[0] es X, corner[1] es Y
                    marker.corners.push_back(cv::Point2f(corner[0], corner[1]));
                }

                markersList.push_back(marker);
            }
        }
    } catch (json::parse_error& e) {
        std::cerr << "Error al parsear JSON: " << e.what() << std::endl;
    }

    return markersList;
}

