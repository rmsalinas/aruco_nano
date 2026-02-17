/**
 * @file aruco_nano.h
 * @brief ArucoNano: A lightweight, header-only ArUco marker detection library.
 * @version 8.0
 *
 * DESCRIPTION:
 * ArucoNano compresses robust ArUco marker detection into a single header file
 * (< 500 lines) that you can simply drop into your project. It is designed for
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
    aruco_nano::Params params;
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
    aruco_nano::Params params;
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
 *
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
#define ArucoNanoVersion 8
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/objdetect/aruco_detector.hpp> // We extract the dictionary from here
#include <vector>

#include <opencv2/highgui.hpp>
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
    //draws it in a image
    inline void draw(cv::Mat &image,const cv::Scalar color=cv::Scalar(0,0,255))const;
    //given the camera params, returns the Rvec,Tvec indicating the pose of the marker wrt the camer. This is just a call to cv::solvePnp using the  cv::SOLVEPNP_IPPE  method
    inline std::pair<cv::Mat,cv::Mat> estimatePose(cv::Mat cameraMatrix,cv::Mat distCoeffs,double markerSize=1.0f)const;
};
struct Params {
    int boxFilterSize=15,thres=3; //values for adaptive thresholding
    int minSize=10;//minimum size of a contour side to be considered as a marker candidate
    int maxAttemptsPerCandidate=5;//number of attempts to identify a candidate by slightly altering the corners
    cv::aruco::Dictionary dict= cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    // [1,n] ; maximum number of times a contour can revisit any of its pixels (1 is the minimum which is the starting point)
    //if you set a high value (std::numeric_limits<int>::max()) the algorithm behaves as the normal moore contour tracer
    int maxTimesRevisited=4;
    /// number of bits of the marker border, i.e. marker border width (default 1).
    int  markerBorderBits=1; //i do not see this useful. all dicts have 1 border bit but its used in opencv  aruco and I keep it here
    double errorCorrectionRate=0;//The default 0.6 value in aruco opencv is very dangerous. It causes many false positives.
    double maxErroneousBitsInBorderRate=0;//maximum rate of erroneous bits in the border. Default 0 means no error allowed.
    bool detectInvertedMarker=false;//if the markers are printed in white over black background
};
/** @brief The MarkerDetector class is detecting the markers in the image passed */
class MarkerDetector{
public:
    // The only function you need to call
    static inline std::vector<Marker> detect(const cv::Mat &img, const Params &params=Params());
private:
    static inline Marker sort( const  Marker &marker);
    static inline float  getSubpixelValue(const cv::Mat &im_grey,const cv::Point2f &p);
    static inline int   getMarkerId(  cv::Mat  candidateBits,int &idx, int &nrotations, const Params &params);
    static inline int    perimeter(const std::vector<cv::Point2f>& a);
    static inline int isInto(const std::vector<cv::Point2f> &a, const std::vector<cv::Point2f> &b) ;
    static std::vector<std::vector<cv::Point>> visitedAwareTracingContour(cv::Mat &padded, size_t minSize = 1,int maxTimesRevisited=4) ;
    static int getBorderErrors(const cv::Mat &bits, int markerSize, int borderSize) ;
    static void thres255Adaptive(cv::Mat &in,cv::Mat &out,int off=2,int thres=5);
};
/** @brief ArucoDetector mimics OpenCV API */
class ArucoDetector{
public:
    ArucoDetector(){}
    ArucoDetector(const cv::aruco::Dictionary &dict, const Params &params= {}  ){
        _params=params;
        _params.dict=dict;
    }
    void detectMarkers (cv::InputArray image, cv::OutputArrayOfArrays corners, cv::OutputArray ids) const;
private:
    Params _params;
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
void ArucoDetector::detectMarkers(cv::InputArray _image, cv::OutputArrayOfArrays _corners, cv::OutputArray _ids ) const   {
    cv::Mat image = _image.getMat();
    CV_Assert(!image.empty());

    // 1. Run internal detection logic
    std::vector<Marker> markers = MarkerDetector::detect(image,_params);

    // 2. Unpack results into OutputArrayOfArrays
    copyVector2Output(markers, _corners);


    // 3. Assign to output ids
    std::vector<int> idsVec;
    idsVec.reserve(markers.size());
    for (const auto& m : markers) idsVec.push_back(m.id);

    // Allocate and copy IDs
    _ids.create((int)idsVec.size(), 1, CV_32SC1);
    cv::Mat(idsVec).copyTo(_ids);
}
std::vector<Marker>  MarkerDetector::detect(const cv::Mat &img, const Params &params ){
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
    cv::Mat bits(params.dict.markerSize+2,params.dict.markerSize+2,CV_8UC1),bitadaptive(params.dict.markerSize+2,params.dict.markerSize+2,CV_8UC1);
    ///////////////// for each contour, approx to a rectangle and check bits inside
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
            if(i==2){ // if not working the first time, try this time adaptive threshold into the bits to improve robustness to lighting
                thres255Adaptive(bits,bitadaptive);
                bitadaptive.copyTo(bits);
            }
            else{
                cv::threshold(bits,bits,0,255,cv::THRESH_OTSU);
            }
            //now, analyze the inner code to see it if is a marker. If so, rotate to have the points properly sorted
            int nrotations=0;
            if(getMarkerId(bits,marker.id,nrotations,params)==0) continue;
            std::rotate(marker.begin(),marker.begin() + 4 - nrotations,marker.end());
        }
        if(marker.id!=-1) DetectedMarkers.push_back(marker);
    }
    /// REMOVAL OF INNER DUPLICATED DETECTIONS OF THE SAME MARKER(INNER AND OUTER BORDER)
    std::sort(DetectedMarkers.begin(), DetectedMarkers.end(),[](const Marker &a,const Marker &b){return a.id<b.id;});
    {
        std::vector<bool> toRemove(DetectedMarkers.size(), false);
        for (int i = 0; i < int(DetectedMarkers.size()) - 1; i++)
        {
            for (int j = i + 1; j < int(DetectedMarkers.size()) && !toRemove[i]; j++)
            {
                if (DetectedMarkers[i].id == DetectedMarkers[j].id )
                {
                    auto res=isInto(DetectedMarkers[i],DetectedMarkers[j]);
                    //std::cout<<DetectedMarkers[i].id<<" "<<DetectedMarkers[j].id<< " res: "<<res<<std::endl;
                    if( res==1)toRemove[i]=true;
                    else if( res==2)toRemove[j]=true;

                }
            }
        }
        //now remove the marked ones
        std::vector<Marker>  DetectedMarkers2;
        for (unsigned int i = 0; i < DetectedMarkers.size(); i++)
            if (!toRemove[i]) DetectedMarkers2.push_back(DetectedMarkers[i]);
        DetectedMarkers=DetectedMarkers2;
    }
    ////// finally subpixel corner refinement
    if(DetectedMarkers.size()>0){
        int halfwsize= 4*float(bwimage.cols)/float(bwimage.cols) +0.5 ;
        std::vector<cv::Point2f> Corners;
        for (const auto &m:DetectedMarkers)
            Corners.insert(Corners.end(), m.begin(),m.end());
        cv::cornerSubPix(bwimage, Corners, cv::Size(halfwsize,halfwsize), cv::Size(-1, -1),cv::TermCriteria( cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005));
        // copy back to the markers
        for (unsigned int i = 0; i < DetectedMarkers.size(); i++)
            for (int c = 0; c < 4; c++) DetectedMarkers[i][c] = Corners[i * 4 + c];
    }
    return DetectedMarkers;//DONE
}
int  MarkerDetector::perimeter(const std::vector<cv::Point2f>& a)
{
    int sum = 0;
    for (size_t i = 0; i < a.size(); i++)
        sum+=cv::norm( a[i]-a[(i + 1) % a.size()]);
    return sum;
}
/**
 * @brief Tries to identify one candidate given the dictionary
 * @return candidate typ. zero if the candidate is not valid,
 *                           1 if the candidate is a black candidate (default candidate)
 *                           2 if the candidate is a white candidate
 */
int MarkerDetector:: getMarkerId(cv::Mat candidateBits, int &idx, int &nrotations, const Params &params){
    uint8_t typ=1;

    if(params.detectInvertedMarker ) candidateBits=~candidateBits;
    // analyze border bits
    int maximumErrorsInBorder =int(params.dict.markerSize * params.dict.markerSize * params.maxErroneousBitsInBorderRate);
    int borderErrors =getBorderErrors(candidateBits, params.dict.markerSize, params.markerBorderBits);
    if(borderErrors > maximumErrorsInBorder) return 0; // border is wrong
    // take only inner bits
    cv::Mat onlyBits =candidateBits.rowRange(params.markerBorderBits,candidateBits.rows - params.markerBorderBits).colRange(params.markerBorderBits, candidateBits.cols - params.markerBorderBits);
    onlyBits/=255;
    // try to indentify the marker
    if(!params.dict.identify(onlyBits, idx, nrotations, params.errorCorrectionRate))
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
std::vector<std::vector<cv::Point>> MarkerDetector::visitedAwareTracingContour(cv::Mat &padded,size_t minSize,int maxTimesRevisited ) {
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
    for (int r = 1; r < rows - 1; ++r) {
        uchar* row_ptr = data + r * step;
        for (int c = 1; c < cols - 1;  ) {
            ////findStartContourPoint
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
            if( c==cols) break;//reached end of row
            if (row_ptr[c] == FOREGROUND ) {// --- 4. Tracing Loop  if is foreground
                buffer.clear();
                int curr_x = c, curr_y = r,search_idx = 1 ;
                uchar* curr_ptr = row_ptr + c,*start_ptr=curr_ptr;
                int ntimesRevisited=0;
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
                if (ntimesRevisited<=maxTimesRevisited && buffer.size() >= minSize) {
                    contours.push_back(buffer);
                }
            }
            c++;//move to next pixel
            ////findEndContourPoint
            if ( row_ptr[c]){
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
