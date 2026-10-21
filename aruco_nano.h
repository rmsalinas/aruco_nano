/**
 * @file aruco_nano.h
 * @brief ArucoNano: A lightweight, header-only ArUco marker detection library.
 * @version 10.0
 * @author Rafael Munoz-Salinas
 * @copyright 2026 Rafael Munoz-Salinas
 *
 * DESCRIPTION:
 * ArucoNano compresses robust ArUco marker detection into a single header file
 * that you can simply drop into your project. It is designed for
 * speed and ease of integration.
 *
 * KEY FEATURES:
 * - Header-only: No linking required, just #include.
 * - High Performance: Uses custom "visited aware tracing" and SIMD optimizations.
 * - OpenCV Compatible: Mimics the cv::aruco API for easy migration.
 * - Customizable: Fine-tune detection via the Params struct (dictionaries, thresholding, etc.).
 *
 * ----------------------------------------------------------------------------
 * QUICK START / USAGE EXAMPLES
 * ----------------------------------------------------------------------------
 *
  //1. BASIC DETECTION (Nano Style): Using DICT_ARUCO_MIP_36h12 by default
  #include <opencv2/highgui.hpp>
  #include "aruco_nano.h"
  int main() {
    cv::Mat image = cv::imread("/path/to/image.png");
    auto markers=aruco_nano::MarkerDetector::detect(image);
    for(const auto &m:markers)
        m.draw(image);
    cv::imwrite("/path/to/saveimage.png",image);
    return 0;
  }

  //2 BASIC DETECTION (Nano Style+Dict selection)
  #include <opencv2/highgui.hpp>
  #include "aruco_nano.h"
  int main() {
    cv::Mat image = cv::imread("/path/to/image.png");
    aruco_nano::DetectorParameters params;
    params.dict=cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000);
    auto markers=aruco_nano::MarkerDetector::detect(image,params);
    for(const auto &m:markers)
        m.draw(image);
    cv::imwrite("/path/to/saveimage.png",image);
    return 0;
  }

  //3. BASIC DETECTION (OpenCV Style):
  #include <opencv2/highgui.hpp>
  #include "aruco_nano.h"
  int main() {
    cv::Mat image = cv::imread("/path/to/image.png");
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    aruco_nano::ArucoDetector detector(dictionary);
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    detector.detectMarkers(image, corners, ids);
    cv::aruco::drawDetectedMarkers(image,corners,ids);
    return 0;
  }

  //4. ADVANCED USAGE (Dict Selection+ Inverted Color Markers(White and Black)+ Pose Estimation):
  #include <opencv2/highgui.hpp>
  #include "aruco_nano.h"
  int main() {
    cv::Mat image = cv::imread("/path/to/image.png");
    aruco_nano::DetectorParameters params;
    params.dict=cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    params.detectInvertedMarker=true;//white markers in black background
    auto markers=aruco_nano::MarkerDetector::detect(image,params);
    cv::Mat camMatrix, distCoeff; // Load these from calibration file
    float markerSize = 0.05f; // 5cm
    for( auto &m:markers){
        auto pose = m.estimatePose(camMatrix, distCoeff, markerSize);
        std::cout<<"Rvec="<<pose.first<<" Tvec="<<pose.second<<std::endl;
    }
    return 0;
  }
 *
 * ----------------------------------------------------------------------------
 * CITATION
 * ----------------------------------------------------------------------------
 * If you use this file in your research, you MUST cite:
 *
 * 1.
 * "Automatic generation and detection of highly reliable fiducial markers under occlusion",
 * S. Garrido-Jurado, R. Muñoz Salinas, F.J. Madrid-Cuevas,M.J. Marín-Jiménez.
 * Pattern Recognition,  vol 47, pages 2280-2292, 2014
 *
 * 2.
 * "Generation of fiducial marker dictionaries using mixed integer linear programming",
 * S. Garrido-Jurado, R. Muñoz Salinas, F.J. Madrid-Cuevas, R. Medina-Carnicer,
 * Pattern Recognition: vol 51,pages 481-491, 2016.
 *
 * 3.
 * "Speeded up detection of squared fiducial markers", Francisco J.Romero-Ramirez,
 * Rafael Muñoz-Salinas, Rafael Medina-Carnicer. Image and Vision Computing,
 * vol 76, pages 38-47, year 2018.
 *
 * 4. Pending publication.
 *
 * You can freely use the code in your commercial products.
 *
 * ----------------------------------------------------------------------------
 * CHANGELOG
 * ----------------------------------------------------------------------------
 * Version 5: Added support for AprilTag 36h11.
 * Version 6: Increased adaptive threshold window; removed obfuscation.
 * Version 8: MAJOR OVERHAUL
 * - Namespace changed to aruco_nano.
 * - New contour tracing algorithm "visited aware tracing" (faster than cv::findContours).
 * - Added Params struct for fine-grained control.
 * - Proper removal of duplicates.
 * - Added OpenCV Dictionary support.
 * - Added wrapper class ArucoDetection for OpenCV API compatibility.
 * Version 9
 * - maxTimesRevisited changed to floating value (0,1), so it now is a percentage of the contour's lenght
 * Version 10
 * - Added support for multiple dictionaries
 * - Params renamed to DetectorParameters be more like OpenCV
 * - Added output of all candidates to be more like OpenCV
 * ----------------------------------------------------------------------------
 * LICENSE
 * ----------------------------------------------------------------------------
 *  Copyright (c) 2026 University of Cordoba
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once
#define ArucoNanoVersion 10
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/objdetect/aruco_detector.hpp> // We extract the dictionary from here
#include <vector>

namespace aruco_nano  {
/**
 * @brief The Marker class is a marker detectable by the library
 * It is a vector where each corner is a corner of the detected marker
 */
class Marker : public std::vector<cv::Point2f>
{
public:
    // id of  the marker
    int id=-1;
    //id of the dict
    int dict=-1;
    //draws it in a image
    inline void draw(cv::Mat &image,const cv::Scalar color=cv::Scalar(0,0,255))const;
    //given the camera params, returns the Rvec,Tvec indicating the pose of the marker wrt the camer. This is just a call to cv::solvePnp using the  cv::SOLVEPNP_IPPE  method
    inline std::pair<cv::Mat,cv::Mat> estimatePose(cv::Mat cameraMatrix,cv::Mat distCoeffs,double markerSize=1.0f)const;
};
struct DetectorParameters {
    int boxFilterSize=15,thres=3; //values for adaptive thresholding
    int minSize=10;//minimum size of a contour side to be considered as a marker candidate
    int maxAttemptsPerCandidate=5;//number of attempts to identify a candidate by slightly altering the corners
    std::vector<cv::aruco::Dictionary> dicts= {cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12)};
    // [0,1] ; maximum number of times a contour can revisit any of its pixels (1 is the minimum which is the starting point)
    //if you set a high value (std::numeric_limits<int>::max()) the algorithm behaves as the normal moore contour tracer
    float maxTimesRevisited=0.05; //1 equals tradional algo,
    /// number of bits of the marker border, i.e. marker border width (default 1).
    float  markerBorderBits=1; //i do not see this useful. all dicts have 1 border bit but its used in opencv  aruco and I keep it here
    double errorCorrectionRate=0;//The default 0.6 value in aruco opencv is very dangerous. It causes many false positives.
    double maxErroneousBitsInBorderRate=0;//maximum rate of erroneous bits in the border. Default 0 means no error allowed.
    bool detectInvertedMarker=false;//if the markers are printed in white over black background
};
/** @brief The MarkerDetector class is detecting the markers in the image passed */
class MarkerDetector{
public:
    // The only function you need to call
    static inline std::vector<Marker> detect(const cv::Mat &img, const DetectorParameters &params=DetectorParameters(),std::vector<Marker> *candidatesOut=nullptr);
private:
    static inline Marker sort( const  Marker &marker);
    static inline float  getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p);
    static inline void refineCornersEdges(const cv::Mat &im_grey, std::vector<Marker> &markers, const DetectorParameters &params);
    static inline int   getMarkerId(  cv::Mat  candidateBits,int &idx, int &nrotations, const DetectorParameters &params,int dictIndex);
    static inline int isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) ;
    static std::vector<std::vector<cv::Point>> visitedAwareTracingContour(cv::Mat &padded, size_t minSize = 1,float maxRevisited=0.1) ;
    static int getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) ;
    static void thres255Adaptive(cv::Mat &in,cv::Mat &out,int off=2,int thres=5);
};
/** @brief ArucoDetector mimics OpenCV API */
class ArucoDetector{
public:
    ArucoDetector(){}
    ArucoDetector(const cv::aruco::Dictionary &dict, const DetectorParameters &params= {}  ){
        _params=params;
        _params.dicts.push_back(dict);
    }
    ArucoDetector(const std::vector<cv::aruco::Dictionary> &dicts, const DetectorParameters &params= {}  ){
        _params=params;
        _params.dicts=dicts;
    }
    void detectMarkers (cv::InputArray image, cv::OutputArrayOfArrays corners, cv::OutputArray ids,  cv::OutputArrayOfArrays rejectedImgPoints = cv::noArray()) const;
    void detectMarkersMultiDict(cv::InputArray image, cv::OutputArrayOfArrays corners, cv::OutputArray ids,
                                cv::OutputArrayOfArrays rejectedImgPoints = cv::noArray(), cv::OutputArray dictIndices = cv::noArray()) const;
private:
    DetectorParameters _params;
    void copyVector2Output(std::vector<Marker> &vec, cv::OutputArrayOfArrays out )const ;
};
namespace _private {
struct Homographer{
    Homographer(const std::vector<cv::Point2f> & out ){
        std::vector<cv::Point2f>  in={cv::Point2f(0,0),cv::Point2f(1,0),cv::Point2f(1,1),cv::Point2f(0,1)};
        H=cv::getPerspectiveTransform(in, out);
    }
    cv::Point2f operator()(const cv::Point2f &p){
        double *m=H.ptr<double>(0);
        double c=m[6]*p.x+m[7]*p.y+m[8];
        return cv::Point2f((m[0]*p.x+m[1]*p.y+m[2])/c,(m[3]*p.x+m[4]*p.y+m[5])/c);
    }
    cv::Mat H;
};
}
//Marker intersection. Tells the marker with most corners into another. 0 if no intersection or tie
int MarkerDetector::isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) {
    // Lambda for point-in-polygon test (Ray Casting)
    auto countInside = [](const std::vector<cv::Point2f>& source, const std::vector<cv::Point2f>& target) -> int {
        int count = 0;
        for (const auto& pt : source) {
            bool inside = false;
            // Fixed 4-side loop logic
            for (int i = 0, j = 3; i < 4; j = i++) {
                if (((target[i].y > pt.y) != (target[j].y > pt.y)) &&
                    (pt.x < (target[j].x - target[i].x) * (pt.y - target[i].y) / (target[j].y - target[i].y) + target[i].x)) {
                    inside = !inside;
                }
            }
            if (inside) count++;
        }
        return count;
    };
    // Count how many corners of A are in B
    int aInB = countInside(a, b);
    // Count how many corners of B are in A
    int bInA = countInside(b, a);
    // Rule 1: Must contain at least one corner
    if (aInB == 0 && bInA == 0) return 0;
    // Rule 2: Compare counts
    if (aInB > bInA) return 1;
    if (bInA > aInB) return 2;
    // Default: Tie or no relative dominance
    return 0;
}

void ArucoDetector::detectMarkersMultiDict(cv::InputArray _image, cv::OutputArrayOfArrays _corners, cv::OutputArray _ids,
                                           cv::OutputArrayOfArrays _rejectedImgPoints , cv::OutputArray _dictIndices ) const {
    cv::Mat image = _image.getMat();
    CV_Assert(!image.empty());
    std::vector<Marker> rejectedImgPointsVec;

    // 1. Run internal detection logic
    std::vector<Marker> markers = MarkerDetector::detect(image,_params,&rejectedImgPointsVec);

    // 2. Unpack results into OutputArrayOfArrays
    copyVector2Output(markers, _corners);
    if( _rejectedImgPoints.needed())
        copyVector2Output(rejectedImgPointsVec, _rejectedImgPoints);


    // 3. Assign to output ids
    std::vector<int> idsVec;
    idsVec.reserve(markers.size());
    for (const auto& m : markers) idsVec.push_back(m.id);

    // Allocate and copy IDs
    _ids.create((int)idsVec.size(), 1, CV_32SC1);
    cv::Mat(idsVec).copyTo(_ids);

    //4. Assign dict indices
    std::vector<int> dictIndicesVec;
    dictIndicesVec.reserve(markers.size());
    for (const auto& m : markers) dictIndicesVec.push_back(m.dict);
    if(_dictIndices.needed()) {
        _dictIndices.create((int)dictIndicesVec.size(), 1, CV_32SC1);
        cv::Mat(dictIndicesVec).copyTo(_dictIndices);
    }

}

void ArucoDetector::detectMarkers(cv::InputArray _image, cv::OutputArrayOfArrays _corners, cv::OutputArray _ids,cv::OutputArrayOfArrays _rejectedImgPoints ) const   {
    cv::Mat image = _image.getMat();
    CV_Assert(!image.empty());
    std::vector<Marker> rejectedImgPointsVec;

    // 1. Run internal detection logic
    std::vector<Marker> markers = MarkerDetector::detect(image,_params,&rejectedImgPointsVec);

    // 2. Unpack results into OutputArrayOfArrays
    copyVector2Output(markers, _corners);
    if( _rejectedImgPoints.needed())
        copyVector2Output(rejectedImgPointsVec, _rejectedImgPoints);


    // 3. Assign to output ids
    std::vector<int> idsVec;
    idsVec.reserve(markers.size());
    for (const auto& m : markers) idsVec.push_back(m.id);

    // Allocate and copy IDs
    _ids.create((int)idsVec.size(), 1, CV_32SC1);
    cv::Mat(idsVec).copyTo(_ids);
}
std::vector<Marker>  MarkerDetector::detect(const cv::Mat &img, const DetectorParameters &params,std::vector<Marker> *candidatesOut){
    cv::Mat bwimage,thresImage;
    std::vector<Marker> DetectedMarkers;
    //first, convert to bw
    if(img.channels()==3)
        cv::cvtColor(img,bwimage,cv::COLOR_BGR2GRAY);
    else bwimage=img;
    /////////////////// Adaptive Threshold to detect border
    //    cv::adaptiveThreshold(bwimage, thresImage, 255.,cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, params.boxFilterSize, params.Thres);
    //this method is achieves a ~1.5 speed up
    cv::boxFilter( bwimage, thresImage, bwimage.type(), cv::Size(params.boxFilterSize, params.boxFilterSize),cv::Point(-1,-1), true, cv::BORDER_REPLICATE|cv::BORDER_ISOLATED );
    thresImage=thresImage-bwimage;
    cv::threshold(thresImage, thresImage, params.thres, 255, cv::THRESH_BINARY);
    /////////////////// compute marker candidates by detecting contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Point> approxCurve;
    cv::RNG rand;
    //cv::findContours(thresImage, contours, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    int  minSizeSq=params.minSize*params.minSize,minSize4=4*params.minSize;
    contours=visitedAwareTracingContour(thresImage,minSize4,params.maxTimesRevisited);

    //decide where to store the candidates. If candidatesOut is not null, store there, otherwise use a local variable
    std::vector<Marker> candidateslocal;
    if(candidatesOut!=nullptr) {
        candidatesOut->clear();
    }
    else{
        candidatesOut=&candidateslocal;
    }
    ///////////////// for each contour, approx to a rectangle
    for (unsigned int i = 0; i < contours.size(); i++)
    {
        // can approximate to a convex rect?
        cv::approxPolyDP(contours[i], approxCurve, double(contours[i].size()) * 0.03, true);
        if (approxCurve.size() != 4 || !cv::isContourConvex(approxCurve)) continue;
        //check distance  between corners at least minSize pix
        if(  ((approxCurve[0].x-approxCurve[1].x)*(approxCurve[0].x-approxCurve[1].x) + (approxCurve[0].y-approxCurve[1].y)*(approxCurve[0].y-approxCurve[1].y))<minSizeSq) continue;
        if(  ((approxCurve[1].x-approxCurve[2].x)*(approxCurve[1].x-approxCurve[2].x) + (approxCurve[1].y-approxCurve[2].y)*(approxCurve[1].y-approxCurve[2].y))<minSizeSq) continue;
        if(  ((approxCurve[2].x-approxCurve[3].x)*(approxCurve[2].x-approxCurve[3].x) + (approxCurve[2].y-approxCurve[3].y)*(approxCurve[2].y-approxCurve[3].y))<minSizeSq) continue;
        if(  ((approxCurve[3].x-approxCurve[0].x)*(approxCurve[3].x-approxCurve[0].x) + (approxCurve[3].y-approxCurve[0].y)*(approxCurve[3].y-approxCurve[0].y))<minSizeSq) continue;
        // // add the points
        Marker marker;marker.reserve(4);
        for (int j = 0; j < 4; j++)
            marker.emplace_back( cv::Point2f( approxCurve[j].x,approxCurve[j].y));
        //sort corner in clockwise direction
        marker=sort(marker);
        candidatesOut->push_back(marker);
    }


    //now, for each candidate check bits inside
    int dictIndex=-1;
    for(auto dict:params.dicts){
        std::vector<Marker> currDirMarkerDetected;
        dictIndex++;
        cv::Mat bits(dict.markerSize+2,dict.markerSize+2,CV_8UC1),bitadaptive(dict.markerSize+2,dict.markerSize+2,CV_8UC1);

        for(auto it=candidatesOut->begin();it!=candidatesOut->end();){
            auto marker=*it;

            ////// extract the code. Obtain the intensities of the bits using  homography
            for(int i=0;i<int(params.maxAttemptsPerCandidate) && marker.id==-1;i++){
                //if not first attempt, we may wanna produce small random alteration of the corners
                auto marker2=marker;
                if( i!=0) for(int c=0;c<4;c++) {marker2[c].x+=rand.gaussian(0.75);marker2[c].y+=rand.gaussian(0.75);}//if not first, alter corner location
                _private::Homographer hom(marker2);
                for(int r=0;r<bits.rows;r++){
                    for(int c=0;c<bits.cols;c++){
                        bits.at<uchar>(r,c)=uchar(0.5+getSubpixelValue(bwimage,hom(cv::Point2f(  float(c+0.5) / float(bits.cols) ,  float(r+0.5) / float(bits.rows)  ))));
                    }
                }
                if(i==0){//prevent elements with low entropy from being analyzed
                    double minv=255,maxv=0;
                    cv::minMaxLoc(bits,&minv,&maxv);
                    if( maxv-minv < 10) break;
                }
                if(i==2){ // if not working the first time, try this time adaptive threshold into the bits to improve robustness to lighting
                    thres255Adaptive(bits,bitadaptive);
                    bitadaptive.copyTo(bits);
                }
                else{
                    cv::threshold(bits,bits,0,255,cv::THRESH_OTSU);
                }
                //now, analyze the inner code to see it if is a marker. If so, rotate to have the points properly sorted
                int nrotations=0;
                if(getMarkerId(bits,marker.id,nrotations,params,dictIndex)==0) continue;
                std::rotate(marker.begin(),marker.begin() + 4 - nrotations,marker.end());
            }
            if(marker.id!=-1) {
                marker.dict=dictIndex;
                currDirMarkerDetected.push_back(marker);
                //remove from candidate list
                it=candidatesOut->erase(it);
            }
            else it++;//go to next
        }

        /// REMOVAL OF INNER DUPLICATED DETECTIONS OF THE SAME MARKER(INNER AND OUTER BORDER)
        std::sort(currDirMarkerDetected.begin(), currDirMarkerDetected.end(),[](const Marker &a,const Marker &b){return a.id<b.id;});
        {
            std::vector<bool> toRemove(currDirMarkerDetected.size(), false);
            for (int i = 0; i < int(currDirMarkerDetected.size()) - 1; i++)
            {
                for (int j = i + 1; j < int(currDirMarkerDetected.size()) && !toRemove[i]; j++)
                {
                    if (currDirMarkerDetected[i].id == currDirMarkerDetected[j].id )
                    {
                        auto res=isInto(currDirMarkerDetected[i],currDirMarkerDetected[j]);
                        if( res==1)toRemove[i]=true;
                        else if( res==2)toRemove[j]=true;

                    }
                }
            }
            //now move to DetectedMarkers these not marked for removal
            for (unsigned int i = 0; i < currDirMarkerDetected.size(); i++)
                if (!toRemove[i]) DetectedMarkers.push_back(currDirMarkerDetected[i]);

        }
    }


    ////// finally subpixel corner refinement
    refineCornersEdges(bwimage,DetectedMarkers,params);
    return DetectedMarkers;//DONE
}
/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
int MarkerDetector:: getMarkerId(cv::Mat candidateBits, int &idx, int &nrotations, const DetectorParameters &params,int dictIndex){
    uint8_t typ=1;

    if(params.detectInvertedMarker ) candidateBits=~candidateBits;
    // analyze border bits
    int maximumErrorsInBorder =int(params.dicts[dictIndex].markerSize * params.dicts[dictIndex].markerSize * params.maxErroneousBitsInBorderRate);
    int borderErrors =getBorderErrors(candidateBits, params.dicts[dictIndex].markerSize, params.markerBorderBits);
    if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong
    // take only inner bits
    cv::Mat onlyBits =candidateBits.rowRange(params.markerBorderBits,candidateBits.rows - params.markerBorderBits).colRange(params.markerBorderBits, candidateBits.cols - params.markerBorderBits);
    onlyBits/=255;
    // try to indentify the marker
    if(!params.dicts[dictIndex].identify(onlyBits, idx, nrotations, params.errorCorrectionRate))
        return 0;
    return typ;
}
/**
  * @brief Return number of erroneous bits in border, i.e. number of white bits in border.
  */
int MarkerDetector::getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) {
    int sizeWithBorders = markerSize + 2 * borderSize;
    int totalErrors = 0;
    for(int y = 0; y < sizeWithBorders; y++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(y)[k] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(y)[sizeWithBorders - 1 - k] != 0) totalErrors++;
        }
    }
    for(int x = borderSize; x < sizeWithBorders - borderSize; x++) {
        for(int k = 0; k < borderSize; k++) {
            if(bits.ptr<unsigned char>(k)[x] != 0) totalErrors++;
            if(bits.ptr<unsigned char>(sizeWithBorders - 1 - k)[x] != 0) totalErrors++;
        }
    }
    return totalErrors;
}
float MarkerDetector::getSubpixelValue(const cv::Mat &im_grey, const cv::Point2f &p) {
    // 1. Get integer coordinates
    const int ix = static_cast<int>(p.x);
    const int iy = static_cast<int>(p.y);


    //   Boundary Check: Ensure the 2x2 patch is within limits
    // We check ix+1 and iy+1 because the interpolation looks at the next pixel over.
    if (ix < 0 || iy < 0 || ix >= im_grey.cols - 1 || iy >= im_grey.rows - 1) {
        // Option A: Return a default value
        // Option B: Clamp the point to the nearest valid boundary
        return 0.0f;
    }

    // 2. Get fractional parts
    const float dx = p.x - ix;
    const float dy = p.y - iy;
    // 3. Optimized Pointer Access
    const uchar* ptr = im_grey.ptr<uchar>(iy) + ix;
    const size_t step = im_grey.step;
    // 4. Fetch the four pixels immediately as floats
    const float p00 = static_cast<float>(ptr[0]);        // Top-Left
    const float p01 = static_cast<float>(ptr[1]);        // Top-Right
    const float p10 = static_cast<float>(ptr[step]);     // Bottom-Left
    const float p11 = static_cast<float>(ptr[step + 1]); // Bottom-Right
    // 5. Separable Interpolation (3 Multiplications total)
    const float top = p00 + dx * (p01 - p00);// Interpolate Top Row Horizontally
    const float bot = p10 + dx * (p11 - p10);    // Interpolate Bottom Row Horizontallys
    // Interpolate Vertically between Top and Bottom results
    return top + dy * (bot - top);
}
/**
 * @brief Refines the corners by fitting a line to each marker side and intersecting adjacent lines.
 *
 * For each side, points spread along the whole side (avoiding the corners, which are the least
 * reliable region) are moved along the side normal to the gradient-weighted edge position. A line
 * is fitted to them and the corners are recomputed as the intersections of adjacent lines.
 * Two passes are done: a coarse one starting from the polygon corners and a fine one sampling a
 * narrower band around the lines of the first pass. Markers are processed in parallel.
 */
void MarkerDetector::refineCornersEdges(const cv::Mat &im_grey, std::vector<Marker> &markers, const DetectorParameters &params){
    if (markers.empty()) return;
    auto refineRange = [&](const cv::Range &r) {
        const int nIterations = 2;
        const float stepLength = 0.5f;                     // sampling step along the normal (pixels)
        const int gradOffset = 4;                          // gradient computed between samples 2 px apart
        const float grange = 0.5f * gradOffset * stepLength;
        const float fineRange = 1.5f;                      // search range of the second pass (lines are already close)
        float profile[32];

        for (int mi = r.start; mi < r.end; mi++) {
            Marker &m = markers[mi];
            // search range along the normal: limited to half a bit cell so inner bits do not interfere
            const int nCells = params.dicts[m.dict].markerSize + 2 * int(params.markerBorderBits);
            double side = 0;
            for (int i = 0; i < 4; i++) side += cv::norm(m[i] - m[(i + 1) % 4]);
            side /= 4;
            const float range = std::max(1.5f, std::min(3.f, float(side / nCells) * 0.5f));

            cv::Point2f corners[4] = {m[0], m[1], m[2], m[3]};
            bool valid = true;
            for (int it = 0; it < nIterations && valid; it++) {
                // first pass: coarse (half the samples) from the polygon corners; second pass: all the
                // samples in a narrower band around the lines found in the first pass
                const bool coarse = it == 0;
                const float itRange = coarse ? range : std::min(range, fineRange);
                const int nSteps = int(2 * itRange / stepLength); // edge positions -itRange .. itRange
                const int nProfile = nSteps + gradOffset + 1;     // <= 17 samples
                cv::Vec4d lines[4]; // point (x,y) and unit normal (nx,ny) of the line of side i -> i+1
                for (int edge = 0; edge < 4 && valid; edge++) {
                    const cv::Point2f a = corners[edge], b = corners[(edge + 1) % 4];
                    const cv::Point2f d = b - a;
                    const float len = float(cv::norm(d));
                    if (len < 1) { valid = false; break; }
                    const cv::Point2f n(d.y / len, -d.x / len);

                    // gradient-weighted edge positions; dark->bright and bright->dark transitions are
                    // accumulated separately and the dominant one is used (supports both marker colors).
                    // Weighted line fit statistics relative to 'a' to keep the numbers small.
                    double W[2] = {0, 0}, Sx[2] = {0, 0}, Sy[2] = {0, 0}, Sxx[2] = {0, 0}, Sxy[2] = {0, 0}, Syy[2] = {0, 0};
                    int N[2] = {0, 0};
                    const int nsamples = coarse ? std::max(8, int(len / 16)) : std::max(16, int(len / 8));
                    for (int s = 0; s < nsamples; s++) {
                        const float alpha = float(1 + s) / float(nsamples + 1);
                        const cv::Point2f p0 = d * alpha;  // relative to a
                        const cv::Point2f start = a + p0 + n * (-itRange - grange);
                        const cv::Point2f step = n * stepLength;
                        for (int j = 0; j < nProfile; j++) profile[j] = getSubpixelValue(im_grey, start + step * float(j));
                        float Mn[2] = {0, 0}, Mw[2] = {0, 0};
                        for (int k = 0; k <= nSteps; k++) {
                            const float g = profile[k + gradOffset] - profile[k];
                            const int pol = g > 0 ? 0 : 1;
                            const float w = g * g;
                            Mn[pol] += w * (-itRange + k * stepLength);
                            Mw[pol] += w;
                        }
                        for (int pol = 0; pol < 2; pol++) {
                            if (Mw[pol] <= 0) continue;
                            const double x = p0.x + n.x * Mn[pol] / Mw[pol], y = p0.y + n.y * Mn[pol] / Mw[pol], w = Mw[pol];
                            W[pol] += w; Sx[pol] += w * x; Sy[pol] += w * y;
                            Sxx[pol] += w * x * x; Sxy[pol] += w * x * y; Syy[pol] += w * y * y;
                            N[pol]++;
                        }
                    }
                    const int pol = W[0] >= W[1] ? 0 : 1;
                    if (N[pol] < 3) { valid = false; break; }

                    // weighted total least squares line fit
                    const double Ex = Sx[pol] / W[pol], Ey = Sy[pol] / W[pol];
                    const double Cxx = Sxx[pol] / W[pol] - Ex * Ex, Cxy = Sxy[pol] / W[pol] - Ex * Ey, Cyy = Syy[pol] / W[pol] - Ey * Ey;
                    const double normalTheta = 0.5 * std::atan2(-2 * Cxy, Cyy - Cxx);
                    lines[edge] = cv::Vec4d(a.x + Ex, a.y + Ey, std::cos(normalTheta), std::sin(normalTheta));
                }
                if (!valid) break;

                // corner i+1 is the intersection of line i and line i+1
                cv::Point2f newCorners[4];
                for (int i = 0; i < 4 && valid; i++) {
                    const cv::Vec4d &l1 = lines[i], &l2 = lines[(i + 1) % 4];
                    // n1 . x = n1 . p1  ;  n2 . x = n2 . p2
                    const double c1 = l1[2] * l1[0] + l1[3] * l1[1];
                    const double c2 = l2[2] * l2[0] + l2[3] * l2[1];
                    const double det = l1[2] * l2[3] - l1[3] * l2[2];
                    if (std::fabs(det) < 1e-3) { valid = false; break; }
                    newCorners[(i + 1) % 4] = cv::Point2f(float((c1 * l2[3] - c2 * l1[3]) / det),
                                                          float((l1[2] * c2 - l2[2] * c1) / det));
                }
                // reject the refinement if any corner moves too far from its initial estimate
                for (int i = 0; i < 4 && valid; i++)
                    if (cv::norm(newCorners[i] - m[i]) > 2 * range) valid = false;
                if (valid) std::copy(newCorners, newCorners + 4, corners);
            }
            if (valid) std::copy(corners, corners + 4, m.begin());
        }
    };
    // markers are independent; only parallelize when there are enough to pay the overhead
    if (markers.size() >= 16)
        cv::parallel_for_(cv::Range(0, int(markers.size())), refineRange, double(markers.size()) / 8);
    else
        refineRange(cv::Range(0, int(markers.size())));
}
Marker  MarkerDetector::sort( const  Marker &marker){
    Marker res_marker=marker;
    /// sort the points in anti-clockwise order
    double dx1 = res_marker[1].x - res_marker[0].x;
    double dy1 = res_marker[1].y - res_marker[0].y;
    double dx2 = res_marker[2].x - res_marker[0].x;
    double dy2 = res_marker[2].y - res_marker[0].y;
    double o = (dx1 * dy2) - (dy1 * dx2);
    // if the third point is in the left side, then sort in anti-clockwise order
    if (o < 0.0)  std::swap(res_marker[1], res_marker[3]);
    return res_marker;
}
std::pair<cv::Mat,cv::Mat> Marker::estimatePose(cv::Mat cameraMatrix,cv::Mat distCoeffs,double markerSize) const{
    std::vector<cv::Point3d> markerCorners={ {-markerSize/2.f,markerSize/2.f,0.f},{markerSize/2.f,markerSize/2.f,0.f},{markerSize/2.f,-markerSize/2.f,0.f},{-markerSize/2.f,-markerSize/2.f,0.f}};
    cv::Mat Rvec,Tvec;
    cv::solvePnP(markerCorners,*this,cameraMatrix,distCoeffs,Rvec,Tvec,false,cv::SOLVEPNP_SQPNP);
    return {Rvec,Tvec};
}
void Marker::draw(cv::Mat &in, const cv::Scalar color) const{
    auto _to_string=[](int i){ std::stringstream str;str<<i;return str.str();};
    float flineWidth=  std::max(1.f, std::min(5.f, float(in.cols) / 500.f));
    int lineWidth= round( flineWidth);
    for(int i=0;i<4;i++)
        cv::line(in, (*this)[i], (*this)[(i+1 )%4], color, lineWidth);
    auto p2 =  cv::Point2f(2.f * static_cast<float>(lineWidth), 2.f * static_cast<float>(lineWidth));
    cv::rectangle(in, (*this)[0] - p2, (*this)[0] + p2, cv::Scalar(0, 0, 255, 255), -1);
    cv::rectangle(in, (*this)[1] - p2, (*this)[1] + p2, cv::Scalar(0, 255, 0, 255), lineWidth);
    cv::rectangle(in, (*this)[2] - p2, (*this)[2] + p2, cv::Scalar(255, 0, 0, 255), lineWidth);
    // determine the centroid
    cv::Point2f cent(0, 0);
    for(auto &p:*this) cent+=p;
    cent/=4;
    float fsize=  std::min(3.0f, flineWidth * 0.75f);
    cv::putText(in,_to_string(id), cent-cv::Point2f(10*flineWidth,0),  cv::FONT_HERSHEY_SIMPLEX,fsize,cv::Scalar(255,255,255)-color, lineWidth,cv::LINE_AA);
}
/**
 * @brief Traces the contours of a binary image using our visited aware Tracing algorithm.
 *
 * This function scans a binary image (foreground as 255, background as 0) and
 * finds the external boundaries of all distinct objects.
 */
std::vector<std::vector<cv::Point>> MarkerDetector::visitedAwareTracingContour(cv::Mat &padded, size_t minSize, float maxRevisited ) {
    if (padded.empty() || padded.type() != CV_8UC1) return {};
    // 1. Fast Initialization and Padding
    int rows = padded.rows;
    int cols = padded.cols;
    int32_t step = padded.step;
    uchar* data = padded.data;
    // Fast clear of top and bottom rows
    memset(data, 0, cols);
    memset(data + (rows - 1) * step, 0, cols);
    // Fast clear of left and right columns
    for (int r = 1; r < rows - 1; ++r) {
        uchar* row_ptr = data + r * step;
        row_ptr[0] = 0;
        row_ptr[cols - 1] = 0;
    }
    // 2. Precompute Neighbor Offsets based on image stride This removes the need for Point arithmetic in the loop
    const int offsets[16]={-1,-step-1,-step,-step+1,1,step+1,step,step-1, -1,-step-1,-step,-step+1,1,step+1,step,step-1, };
    // Use static tables to avoid initialization overhead on every call // 8-connectivity offsets relative to center (0,0) // Order: W, NW, N, NE, E, SE, S, SW
    const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 }, dy[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };
    // Pre-allocate results
    std::vector<std::vector<cv::Point>> contours;contours.reserve(2048);
    std::vector<cv::Point> buffer;buffer.reserve(2048);
    const uchar FOREGROUND = 255, BACKGROUND = 0,VISITED = 100;
    // 3. Scanning Loop
    // We iterate using raw pointers for maximum speed
    int rowStep=1;//std::max(1,int(minSize/6));
    for (int r = 1; r < rows - 1; r+=rowStep) {
        uchar* row_ptr = data + r * step;
        for (int c = 1; c < cols - 1;  ) {
            ////findStartContourPoint
            {
#if (CV_SIMD || CV_SIMD_SCALABLE)
                cv::v_uint8 v_zero = cv::vx_setzero_u8();
                for (; c <= cols - cv::VTraits<cv::v_uint8>::vlanes(); c+= cv::VTraits<cv::v_uint8>::vlanes())
                {
                    cv::v_uint8 vmask = (cv::v_ne(cv::vx_load((uchar*)(row_ptr + c)), v_zero));
                    if (v_check_any(vmask))
                    {
                        c += v_scan_forward(vmask);
                        break;
                    }
                }
#endif
                //process last tail
                for (; c < cols && !row_ptr[c]; ++c) ;//last tail
            }
            if( c==cols) break;//reached end of row
            if (row_ptr[c] == FOREGROUND ) {// --- 4. Tracing Loop  if is foreground
                buffer.clear();
                int curr_x = c, curr_y = r,search_idx = 1 ;
                uchar* curr_ptr = row_ptr + c,*start_ptr=curr_ptr;
                size_t ntimesRevisited=0;
                do {
                    buffer.emplace_back(curr_x, curr_y);// Add point
                    *curr_ptr = VISITED;// Mark as visited
                    //showImage(padded);
                    // Search for next foreground pixel. We search 8 neighbors starting from search_idx
                    for (int i = 0; i < 8; ++i) {
                        int idx = search_idx + i; // index into offsets (0..15)
                        uchar* neighbor = curr_ptr + offsets[idx]; // Fast pointer arithmetic
                        if (*neighbor != BACKGROUND) {
                            // Found next boundary pixel
                            curr_ptr = neighbor;
                            int dir = (idx & 7);                             // Update Integer Coordinates using the small static tables(Use modulo 8 to get the distinct direction 0-7)
                            int next_x=curr_x+dx[dir], next_y=curr_y+dy[dir];
                            ntimesRevisited+= int(*neighbor == VISITED);
                            curr_x = next_x;curr_y = next_y;
                            search_idx = (dir + 5) & 7;
                            break;
                        }
                    }
                } while (curr_ptr != start_ptr );
                size_t bufsize=buffer.size();
                if (ntimesRevisited<= float(bufsize)*maxRevisited && bufsize >= minSize) {
                    contours.push_back(buffer);
                }
            }
            c++;//move to next pixel
            ////findEndContourPoint
            if ( row_ptr[c]){
                {
#if (CV_SIMD || CV_SIMD_SCALABLE)

                    cv::v_uint8 v_zero = cv::vx_setzero_u8();
                    for (; c <=  cols - cv::VTraits<cv::v_uint8>::vlanes(); c += cv::VTraits<cv::v_uint8>::vlanes())
                    {
                        cv::v_uint8 vmask = (cv::v_eq(cv::vx_load((uchar*)(row_ptr + c)), v_zero));
                        if (cv::v_check_any(vmask))
                        {
                            c += cv::v_scan_forward(vmask);
                            break;
                        }
                    }
#endif
                }
                for (; c < cols && row_ptr[c]; ++c) ;//last tail
            }
        }
    }
    return contours;
}

void MarkerDetector::thres255Adaptive(cv::Mat &in,cv::Mat &out,int off,int thres){
    cv::boxFilter( in, out, in.type(), cv::Size(off*2+1, off*2+1),
                  cv::Point(-1,-1), true, 4 );

    for(int i = 0; i < in.rows; i++ )
    {
        const uchar* sdata = in.ptr(i);
        uchar* ddata = out.ptr(i);
        for(int j = 0; j < in.cols; j++ )
            ddata[j] = ((ddata[j]-thres  )< sdata[j]) *255;
    }

}
void ArucoDetector::copyVector2Output(std::vector<Marker> &vec, cv::OutputArrayOfArrays out ) const {
    out.create((int)vec.size(), 1, CV_32FC2);
    if(out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            cv::Mat &m = out.getMatRef(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if(out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            cv::UMat &m = out.getUMatRef(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if(out.kind() == cv::_OutputArray::STD_VECTOR_VECTOR){
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            cv::Mat m = out.getMat(i);
            cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}
}