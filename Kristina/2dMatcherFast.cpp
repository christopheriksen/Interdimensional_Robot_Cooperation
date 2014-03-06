/**
 * @file SURF_FlannMatcher
 * @brief SURF detector + descriptor + FLANN Matcher
 * @author A. Huaman
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <math.h>
#include "opencv2/calib3d/calib3d.hpp"
#include <cv.h>
#include <highgui.h>
#include <stdarg.h>
#include <cstdlib>
#include <glob.h>
#include <fstream>
#include <sstream>
#include <dirent.h>

using namespace std;
using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */

class modDMatch: public DMatch {
public:
  //void push_back(const modDMatch& pushee);
private:
  double matchScore_;
  double pixelShift_;
};

bool stringCompare( const string &left, const string &right ){
   for( string::const_iterator lit = left.begin(), rit = right.begin(); lit != left.end() && rit != right.end(); ++lit, ++rit )
      if( tolower( *lit ) < tolower( *rit ) )
         return true;
      else if( tolower( *lit ) > tolower( *rit ) )
         return false;
   if( left.size() < right.size() )
      return true;
   return false;
}

//void modDMatch::push_back(const modDMatch& pushee) {
//
//}

int main( int argc, char** argv )
{
  // Read from a config.txt file
  // The first line should be the image we want to compare
  // The second line should be the directory we're looking for comparsons in
  //    and it should be of the format "/home/robotics/Desktop/pictures/"
  // The third line should specify the file extension of the pictures in the
  //    directory in the format "png"
  // The fourth line should be the directory where we want to save the side-by-
  //    side images with their keypoints and matches drawn and it should be of
  //    the form "/home/robotics/Desktop/FLANNmatches/matches"
  ifstream infile;
  ifstream in_stream;
  vector<string> fileNames;
  string line;
  in_stream.open("/home/robotics/Desktop/config.txt");

  while(!in_stream.eof())
  {
      in_stream >> line;
      fileNames.push_back(line);
  }

  in_stream.close();

  string compareImage = fileNames[0];
  string pictureDirectory = fileNames[1];
  string pictureExtension = fileNames[2];

  pictureDirectory += "*.";
  pictureDirectory += pictureExtension;

  cout << pictureDirectory << endl;

  const char * picDirectory;
  picDirectory = pictureDirectory.c_str();

  // Get the number of files in the directory
  glob_t gl;
  size_t numFiles = 0;
  if(glob(picDirectory, GLOB_NOSORT, NULL, &gl) == 0)
    numFiles = gl.gl_pathc;
  globfree(&gl);
  cout << "Number of files: " << numFiles << endl;

  // Get the file names from the directory
  vector<string> pictureNames;
  const char * picDirec;
  picDirec = fileNames[1].c_str();
  string fileExtention = ".";
  fileExtention += pictureExtension;
  const char * picExt;
  picExt = pictureExtension.c_str();

  DIR* dirFile = opendir( picDirec );
   if ( dirFile ) 
   {
      struct dirent* hFile;
      //errno = 0;
      while (( hFile = readdir( dirFile )) != NULL ) 
      {
         if ( !strcmp( hFile->d_name, "."  )) continue;
         if ( !strcmp( hFile->d_name, ".." )) continue;

         // dirFile.name is the name of the file. Do whatever string comparison 
         // you want here. Something like:
         if ( strstr( hFile->d_name, picExt )) {
          string counts = fileNames[1];
          counts += hFile->d_name;
          pictureNames.push_back(counts);
        }
      } 
      closedir( dirFile );
   }

  sort(pictureNames.begin(), pictureNames.end(), stringCompare);
  for (size_t i = 0; i < pictureNames.size(); ++i) {
    cout << pictureNames[i] << endl;
  }

  std::vector<double> matchScores;
  std::vector<double> maxMatchScores;
  std::vector<double> pixelAverages;
  std::vector<double> bestPixelScores;
  std::vector<modDMatch> top5;
  double maxMatchScore = -1.0;

  if( argc != 1 )
  { readme(); return -1; }

  //std::string imageDirectory = argv[1];
  int count = 0;
  //int numPhotos = atoi(argv[3]);

  for (int k = 0; k < numFiles; ++k) {
    ++count;
    string countString = static_cast<ostringstream*>( &(ostringstream() << count) )->str();

    std::string readString = fileNames[1];
    //readString += "/";
    readString += countString;
    readString += ".";
    readString += pictureExtension;

    Mat img_1 = imread( compareImage, IMREAD_GRAYSCALE );
    Mat img_2 = imread( pictureNames[k], IMREAD_GRAYSCALE );

    cout << "pictureNames[" << k << "]: " << pictureNames[k] << endl;

    if( !img_1.data || !img_2.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

    //-- Step 1: Detect the keypoints using SURF Detector
    //int minHessian = 400;
    int minHessian = 200;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_1, keypoints_2;

    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_1, descriptors_2;

    extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    //printf("-- Max dist : %f \n", max_dist );
    //printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    { if( matches[i].distance < 2*min_dist )
      { good_matches.push_back( matches[i]); }
    }

    for (int i = 0; i < good_matches.size(); ++i) {
      for (int j = i+1; j < good_matches.size(); ++j) {

        //int matchIndex = matchIndices[i]; // Indices correspond to first vector of matches. change to i in the loop over all i
        //DMatch matched = good_matches[matchIndex]; // Get the DMatch at that index
        DMatch matched = good_matches[i];

        int queryIndex = matched.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex = matched.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex = matched.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the first image
        double xVal1 = keypoints_1[queryIndex].pt.x;
        double yVal1 = keypoints_1[queryIndex].pt.y;

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal2 = keypoints_2[trainIndex].pt.x;
        double yVal2 = keypoints_2[trainIndex].pt.y;

        //int matchIndex2 = matchIndices[j]; // Indices correspond to first vector of matches. change to i in the loop over all i
        //DMatch matched2 = good_matches[matchIndex2]; // Get the DMatch at that index
        DMatch matched2 = good_matches[j];

        int queryIndex2 = matched2.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex2 = matched2.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex2 = matched2.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the first image
        double xVal12 = keypoints_1[queryIndex2].pt.x;
        double yVal12 = keypoints_1[queryIndex2].pt.y;

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal22 = keypoints_2[trainIndex2].pt.x;
        double yVal22 = keypoints_2[trainIndex2].pt.y;

        if ((xVal1 == xVal12 && yVal1 == yVal12) || (xVal2 == xVal22 && yVal2 == yVal22)) {
          if (good_matches[i].distance > good_matches[j].distance) {
            good_matches.erase(good_matches.begin() + i);
          } else {
            good_matches.erase(good_matches.begin() + j);
          }
        }
      }
    }

    ////////////// BIDIRECTIONAL MATCHING //////////////

    FlannBasedMatcher matcher2;
    std::vector< DMatch > matches2;
    matcher2.match( descriptors_2, descriptors_1, matches2 );

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_2.rows; i++ )
    { double dist = matches2[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
    //-- PS.- radiusMatch can also be used here.
    std::vector< DMatch > good_matches2;

    for( int i = 0; i < descriptors_2.rows; i++ )
    { if( matches2[i].distance < 2*min_dist )
      { good_matches2.push_back( matches2[i]); }
    }


    std::vector< DMatch > matchedMatches;
    std::vector<int> matchIndices;

    // Check the matches!
    for (int i = 0; i < good_matches.size(); ++i) {
      for (int j = 0; j < good_matches2.size(); ++j) {
        double difference = abs(good_matches[i].distance - good_matches2[j].distance); // how different are the matches?
        if (difference < 0.000003) { // threshold for the differences
          matchedMatches.push_back(good_matches[i]);
          matchIndices.push_back(i);
        }
      }
    }

  #if 0
  for (int i = 0; i < good_matches.size(); ++i) {
      for (int j = 0; j < good_matches2.size(); ++j) {
      
        // Check for lines that intersect!
        //int matchIndex = matchIndices[i]; // Indices correspond to first vector of matches. change to i in the loop over all i
        //DMatch matched = good_matches[matchIndex]; // Get the DMatch at that index
        DMatch matched = good_matches[i];

        int queryIndex = matched.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex = matched.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex = matched.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the first image
        double xVal1 = keypoints_1[queryIndex].pt.x;
        double yVal1 = keypoints_1[queryIndex].pt.y;

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal2 = keypoints_2[trainIndex].pt.x;
        double yVal2 = keypoints_2[trainIndex].pt.y;

        //int matchIndex2 = matchIndices[j]; // Indices correspond to first vector of matches. change to i in the loop over all i
        //DMatch matched2 = good_matches[matchIndex2]; // Get the DMatch at that index
        DMatch matched2 = good_matches2[j];

        int queryIndex2 = matched2.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex2 = matched2.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex2 = matched2.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the first image
        double xVal12 = keypoints_1[queryIndex2].pt.x;
        double yVal12 = keypoints_1[queryIndex2].pt.y;

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal22 = keypoints_2[trainIndex2].pt.x;
        double yVal22 = keypoints_2[trainIndex2].pt.y;

        std::cout << "xVal1: " << xVal1 << std::endl;
        std::cout << "yVal1: " << yVal1 << std::endl;
        std::cout << "xVal2: " << xVal2 << std::endl;
        std::cout << "yVal2: " << yVal2 << std::endl;
        std::cout << "xVal12: " << xVal12 << std::endl;
        std::cout << "yVal12: " << yVal12 << std::endl;
        std::cout << "xVal22: " << xVal22 << std::endl;
        std::cout << "yVal22: " << yVal22 << std::endl;

        if ((abs(xVal1 - xVal22) < 0.05 && abs(yVal1 - yVal22) < 0.05) || (abs(xVal2 - xVal12) < 0.05 && abs(yVal2- yVal12) < 0.05)) {
            matchedMatches.push_back(good_matches[i]);
      }
    }
  }
  #endif

    for (int i = 0; i < matchedMatches.size(); ++i) {
      for (int j = i+1; j < matchedMatches.size(); ++j) {
      
        //int matchIndex = matchIndices[i]; // Indices correspond to first vector of matches. change to i in the loop over all i
        //DMatch matched = good_matches[matchIndex]; // Get the DMatch at that index
        DMatch matched = matchedMatches[i];

        int queryIndex = matched.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex = matched.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex = matched.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the first image
        double xVal1 = keypoints_1[queryIndex].pt.x;
        double yVal1 = keypoints_1[queryIndex].pt.y;

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal2 = keypoints_2[trainIndex].pt.x;
        double yVal2 = keypoints_2[trainIndex].pt.y;

        //int matchIndex2 = matchIndices[j]; // Indices correspond to first vector of matches. change to i in the loop over all i
        //DMatch matched2 = good_matches[matchIndex2]; // Get the DMatch at that index
        DMatch matched2 = matchedMatches[j];

        int queryIndex2 = matched2.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex2 = matched2.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex2 = matched2.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the first image
        double xVal12 = keypoints_1[queryIndex2].pt.x;
        double yVal12 = keypoints_1[queryIndex2].pt.y;

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal22 = keypoints_2[trainIndex2].pt.x;
        double yVal22 = keypoints_2[trainIndex2].pt.y;

        if ((xVal1 == xVal12 && yVal1 == yVal12) || (xVal2 == xVal22 && yVal2 == yVal22)) {
          if (matchedMatches[i].distance > matchedMatches[j].distance) {
            matchedMatches.erase(matchedMatches.begin() + i);
          } else {
            matchedMatches.erase(matchedMatches.begin() + j);
          }
        }
      }
    }

    /////////////// FIND WHERE WE THINK THE SCENE IS ///////////////

    //-- Localize the object from img_1 in img_2
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < matchedMatches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints_1[ matchedMatches[i].queryIdx ].pt );
      scene.push_back( keypoints_2[ matchedMatches[i].trainIdx ].pt );
    }

    std::vector< DMatch > inFrame;

    cout << "obj.size() " << obj.size() << endl;

    // findHomography needs at least 4 points in order to do its calculation
    if (obj.size() > 3) {
      
      Mat H = findHomography( obj, scene, RANSAC );

      // Get the corners of what we think is the object
      std::vector<Point2f> obj_corners(4);
      obj_corners[0] = Point(0,0); obj_corners[1] = Point( img_1.cols, 0 );
      obj_corners[2] = Point( img_1.cols, img_1.rows ); obj_corners[3] = Point( 0, img_1.rows );
      std::vector<Point2f> scene_corners(4);

      perspectiveTransform(obj_corners, scene_corners, H);

      // Find the minimum x and y coordinates
      float minX = min(scene_corners[0].x, scene_corners[3].x);
      float maxX = max(scene_corners[1].x, scene_corners[2].x);
      float minY = min(scene_corners[0].y, scene_corners[1].y);
      float maxY = max(scene_corners[2].y, scene_corners[3].y);

      if (minX > scene_corners[1].x || minX > scene_corners[2].x) {
        minX = 0;
      }
      if (minY > scene_corners[0].y || minY > scene_corners[1].y) {
        minY = 0;
      }
      if (maxX < scene_corners[3].x || maxX < scene_corners[0].x) {
        maxX = 0;
      }
      if (maxY < scene_corners[0].y || maxY < scene_corners[1].y) {
        maxY = 0;
      }

    //std::cout << "new corner 1: " << scene_corners[0] << std::endl;
    //std::cout << "new corner 2: " << scene_corners[1] << std::endl;
    //std::cout << "new corner 3: " << scene_corners[2] << std::endl;
    //std::cout << "new corner 4: " << scene_corners[3] << std::endl;

      for (int i = 0; i < matchedMatches.size(); ++i) {
        DMatch matched = matchedMatches[i];

        int queryIndex = matched.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex = matched.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex = matched.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal2 = keypoints_2[trainIndex].pt.x;
        double yVal2 = keypoints_2[trainIndex].pt.y;

        if ( (xVal2 < (double)maxX) && (xVal2 > (double)minX) && (yVal2 < (double)maxY) && (yVal2 > (double)minY) ) {
          inFrame.push_back(matchedMatches[i]);
        }
      }
    } else {
        for (int i = 0; i < matchedMatches.size(); ++i) {
          //inFrame.push_back(matchedMatches[i]);
        }
    }

    //for (size_t i = 0; i < inFrame.size(); ++i) {
      //std::cout << "x: " << keypoints_1[inFrame[i].trainIdx].pt.x << std::endl;
      //std::cout << "y: " << keypoints_1[inFrame[i].trainIdx].pt.y << std::endl;
    //}

  #if 0
    ////////////// FIND INTERSECTING LINES //////////////

    std::vector< DMatch > nonIntersecting;
    std::vector< int > intersects;

    double xCalc;
    double yCalc;

    for (int i = 0; i < inFrame.size(); ++i) {
      for (int j = 0; j < inFrame.size(); ++j) {
      
        // Check for lines that intersect!
        //int matchIndex = matchIndices[i]; // Indices correspond to first vector of matches. change to i in the loop over all i
        //DMatch matched = good_matches[matchIndex]; // Get the DMatch at that index
        DMatch matched = inFrame[i];

        int queryIndex = matched.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex = matched.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex = matched.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the first image
        double xVal1 = keypoints_1[queryIndex].pt.x;
        double yVal1 = keypoints_1[queryIndex].pt.y;

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal2 = keypoints_2[trainIndex].pt.x;
        double yVal2 = keypoints_2[trainIndex].pt.y;

        //int matchIndex2 = matchIndices[j]; // Indices correspond to first vector of matches. change to i in the loop over all i
        //DMatch matched2 = good_matches[matchIndex2]; // Get the DMatch at that index
        DMatch matched2 = inFrame[j];

        int queryIndex2 = matched2.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
        int trainIndex2 = matched2.trainIdx; // train descriptor index (second image descriptors)
        int trainImgIndex2 = matched2.imgIdx; // train image index

        // Get x and y coordinates of a particular keypoint from the first image
        double xVal12 = keypoints_1[queryIndex2].pt.x;
        double yVal12 = keypoints_1[queryIndex2].pt.y;

        // Get x and y coordinates of a particular keypoint from the second image
        double xVal22 = keypoints_2[trainIndex2].pt.x;
        double yVal22 = keypoints_2[trainIndex2].pt.y;

        //E = xval12, yval12
        //F = xval22, yval22
        //P = xval1, yval1
        //Q = xval2, yval2

        // calculate the cross products for x and y of the first two points based
        // on the line segment formed by the second two points
        xCalc = (xVal22 - xVal12) * (yVal1 - yVal22) - (yVal22 - yVal12) * (xVal1 - xVal22);
        yCalc = (xVal22 - xVal12) * (yVal2 - Vyal22) - (yVal22 - yVal12) * (xVal2 - xVal22);

        // std::cout << "object keypoint: " << keypoints_1[queryIndex2].pt << std::endl;
        // std::cout << "scene keypoint: " << keypoints_2[trainIndex2].pt << std::endl;
        // std::cout << "xCalc: " << xCalc << std::endl;
        // std::cout << "yCalc: " << yCalc << std::endl;

        if ((xCalc > 0 && yCalc > 0) || (xCalc < 0 && yCalc < 0) || (xCalc == 0 && yCalc == 0)) {
          intersects.push_back(0); // this line combination doesn't cause intersection
        } 
        else {
          intersects.push_back(1); // this line combination causes intersection
        }
      }

      int count = 0;
      for (int k = 0; k != intersects.size(); ++k) {
        if (intersects[k] == 1) break; // the line we're looking at intersected another line somewhere
        ++count; // keep track of the number of nonintersections
      }

      if (count == intersects.size()) {
        nonIntersecting.push_back(inFrame[i]); // the line didn't intersect anything else, so we can plot it
      }
      intersects.clear(); // clear our intersection vector and use it again
    }
  #endif
    //-- Draw only "good" matches
    Mat img_matches;

    drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS ); // DRAW_RICH_KEYPOINTS );

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if 0
  //-- Localize the object from img_1 in img_2
  std::vector<Point2f> obj2;
  std::vector<Point2f> scene2;

  for( size_t i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj2.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
    scene2.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
  }

  Mat H2 = findHomography( obj2, scene2, RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners2(4);
  obj_corners2[0] = Point(0,0); obj_corners2[1] = Point( img_1.cols, 0 );
  obj_corners2[2] = Point( img_1.cols, img_1.rows ); obj_corners2[3] = Point( 0, img_1.rows );
  std::vector<Point2f> scene_corners2(4);

  perspectiveTransform( obj_corners2, scene_corners2, H2 );

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  Point2f offset( (float)img_1.cols, 0);
  line( img_matches, scene_corners2[0] + offset, scene_corners2[1] + offset, Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners2[1] + offset, scene_corners2[2] + offset, Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners2[2] + offset, scene_corners2[3] + offset, Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners2[3] + offset, scene_corners2[0] + offset, Scalar( 0, 255, 0), 4 );

  //imshow( "Good Matches & Object detection", img_matches );
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Draw Keypoints for just one image
    //drawKeypoints(img_2, keypoints_2, img_matches, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //-- Show detected matches
    //imshow( "Good Matches", img_matches );

    //for( int i = 0; i < (int)inFrame.size(); i++ )
    //{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, inFrame[i].queryIdx, inFrame[i].trainIdx ); }

    ///////////////// SCORE MATCHES /////////////////

    double matchScore = 0.0;
    double pixelMatchScore = 0.0;
    double scalePixels = 0.0;

    // Check pixel "shift distance"
    // Loop through all match pairs in the image
    std::vector<double> pixelMatchScores;
    for (int i = 0; i < inFrame.size(); ++i) {
      
      DMatch matched = inFrame[i];

      int queryIndex = matched.queryIdx; // query descriptor index (first image descriptors -- rows correspond to keypoint vector indices?)
      int trainIndex = matched.trainIdx; // train descriptor index (second image descriptors)
      int trainImgIndex = matched.imgIdx; // train image index

      // Get x and y coordinates of a particular keypoint from the first image
      double xVal1 = keypoints_1[queryIndex].pt.x;
      double yVal1 = keypoints_1[queryIndex].pt.y;

      // Get x and y coordinates of a particular keypoint from the second image
      double xVal2 = keypoints_2[trainIndex].pt.x;
      double yVal2 = keypoints_2[trainIndex].pt.y;

      double xShift = abs(xVal1 - xVal2);
      double yShift = abs(yVal1 - yVal2);

      scalePixels = sqrt(pow(xShift, 2) + pow(yShift, 2));
      pixelMatchScores.push_back(scalePixels);
      //inFrame[i].pixelShift_ = scalePixels;

      //cout << "scalePixels: " << scalePixels << endl;

      //if (scalePixels == 0) {
        //matchScore += 0;
      //} else {
        //matchScore += 100000.0/scalePixels;
      //}

      //cout << "pixelMatchScore: " << pixelMatchScore/1000.0 << endl;
    }

    double totalPixels = 0.0;
    for (size_t i = 0; i < pixelMatchScores.size(); ++i) {
      totalPixels += 10000.0/pixelMatchScores[i];
    }

    if (pixelMatchScores.size() == 0) {
      pixelAverages.push_back(0);
      bestPixelScores.push_back(0);
    } else {
      pixelAverages.push_back(totalPixels);///(pixelMatchScores.size()));
      bestPixelScores.push_back(totalPixels);///(pixelMatchScores.size()));
    }

    //scalePixels = scalePixels/1000.0;

    cout << matchScore << endl;

    for (size_t i = 0; i < inFrame.size(); ++i) {
      double dist = (double)(inFrame[i].distance);
      double pointScore = 1.0/dist;
      matchScore += pointScore;
      std::cout << "distance: " << dist << std::endl;
      std::cout << "pointScore: " << pointScore << std::endl;
    }
    std::cout << "matchScore for " << k << ": " << matchScore << std::endl;

    if (matchScore > maxMatchScore) {
      maxMatchScore = matchScore;
    //  maxMatchScores.insert(maxMatchScores.begin(),maxMatchScore);
    }

    matchScores.push_back(matchScore);
    maxMatchScores.push_back(matchScore);

    //waitKey(0);

    ///////////////// SAVE COMPARISONS /////////////////

    // Write the image
    vector<int> compression_params;
    compression_params.push_back(3);
    std::string writeString = fileNames[3]; //"/home/robotics/Desktop/FLANNmatches/matches";
    writeString += countString;
    writeString += ".png";
    imwrite(writeString, img_matches, compression_params);

    ///////////////////// END //////////////////////
}
  cout << "top5MatchScores" << endl;
  sort(maxMatchScores.begin(), maxMatchScores.end());
  reverse(maxMatchScores.begin(), maxMatchScores.end());
  std::vector<double> top5MatchScores;

  for (size_t i = 0; i < 5; ++i) {
    top5MatchScores.push_back(maxMatchScores[i]);
    cout << top5MatchScores[i] << endl;
  }

  cout << "maxMatchScores" << endl;
  for(size_t i = 0; i < maxMatchScores.size(); ++i) {
    cout << maxMatchScores[i] << endl;
  }

  cout << "top5MatchScores size: " << top5MatchScores.size() << endl;

  vector<int> compression_params;
  compression_params.push_back(3);
  std::string top5FileRead;
  std::vector<std::string> top5Read;
  std::vector<modDMatch> preliminaryTop5;

#if 0

  int placeHolder = 0;
  while (top5Read.size() < 5) {
  //while (preliminaryTop5.size() < 5) {
    for (size_t i = 0; i < matchScores.size(); ++i) {
      top5FileRead = fileNames[3]; //"/home/robotics/Desktop/FLANNmatches/matches";
      if (matchScores[i] == maxMatchScores[placeHolder]) {
        //top5.push_back(inFrame[i]);
        #if 0
        cout << "matchScores[" << i << "]: " << matchScores[i] << endl;
        cout << "maxMatchScores[" << placeHolder << "]: " << maxMatchScores[placeHolder] << endl;
        #endif
        ++placeHolder;
        //preliminaryTop5.push_back(inFrame[i]);
        
        string placeString = static_cast<ostringstream*>( &(ostringstream() << i + 1) )->str();
        top5FileRead += placeString;
        top5FileRead += ".png";
        top5Read.push_back(top5FileRead);
        
        //string placeString = static_cast<ostringstream*>( &(ostringstream() << placeHolder) )->str();
        //std::string writeString2 = "/home/robotics/Desktop/top5/top";
        //writeString2 += placeString;
        //writeString2 += ".png";
        //imwrite(writeString2, img_matches, compression_params);
      }
    }
  }
#endif
//#if 0
  // Now check the pixel shifts for each image
  // Smaller pixel shifts are better
  sort(bestPixelScores.begin(), bestPixelScores.end());
  reverse(bestPixelScores.begin(), bestPixelScores.end());

  std::vector<double> nonzeroPixelScores;

  // Pixel shifts of 0 are non-matches for the image, so we remove them
  for (size_t i = 0; i < bestPixelScores.size(); ++i) {
    //cout << i << endl;
    if (bestPixelScores[i] != 0) {
      nonzeroPixelScores.push_back(bestPixelScores[i]);
      //bestPixelScores.erase(bestPixelScores.begin() + i);
      cout << i << " " << bestPixelScores[i] << endl;
    }
  }

  bool plusPixelPlaceHolder = false;

  int pixelPlaceHolder = 0;
  while (top5Read.size() < 5) {
   // cout << "1" << endl;
    for (size_t i = 0; i < pixelAverages.size(); ++i) {
     // cout << "2" << endl;
      for (size_t j = 0; j < top5MatchScores.size(); ++j) {
       // cout << "3" << endl;
        if (matchScores[i] == top5MatchScores[j]) {// && pixelAverages[i] == bestPixelScores)
          top5FileRead = fileNames[3];
          if(pixelAverages[i] == nonzeroPixelScores[pixelPlaceHolder]) {
            ++pixelPlaceHolder;
            plusPixelPlaceHolder = true;
            //cout << "pixelPlaceHolder: " << pixelPlaceHolder << endl;
            string placeString = static_cast<ostringstream*>( &(ostringstream() << i + 1) )->str();
            top5FileRead += placeString;
            top5FileRead += ".png";
            top5Read.push_back(top5FileRead);
          }
        }
      }
      
    }
    if (plusPixelPlaceHolder == false) {
        ++pixelPlaceHolder;
      } else {
        plusPixelPlaceHolder = false;
      }
  }
//#endif
  std::cout << "Max Match Score: " << maxMatchScore << std::endl;

  cout << "matchScores" << endl;
  for (int i = 0; i < matchScores.size(); ++i) {
    cout << i+1 << ": " << matchScores[i] << endl;
  }
  //#if 0
  cout << "pixelAverages" << endl;
  for (int i = 0; i < pixelAverages.size(); ++i) {
    cout << i+1 << ": " << pixelAverages[i] << endl;
  }
  //#endif
  cout << "done printing scores" << endl;
  
  cout << top5Read[0] << endl;
  cout << top5Read[1] << endl;
  cout << top5Read[2] << endl;
  cout << top5Read[3] << endl;
  cout << top5Read[4] << endl;

  const char * im1;
  im1 = top5Read[0].c_str();
  const char * im2;
  im2 = top5Read[1].c_str();
  const char * im3;
  im3 = top5Read[2].c_str();
  const char * im4;
  im4 = top5Read[3].c_str();
  const char * im5;
  im5 = top5Read[4].c_str();

  //cout << "top5Read size: " << top5Read.size() << endl;

  IplImage *img1 = cvLoadImage(im1, CV_LOAD_IMAGE_UNCHANGED);
  IplImage *img2 = cvLoadImage(im2, CV_LOAD_IMAGE_UNCHANGED);
  IplImage *img3 = cvLoadImage(im3, CV_LOAD_IMAGE_UNCHANGED);
  IplImage *img4 = cvLoadImage(im4, CV_LOAD_IMAGE_UNCHANGED);
  IplImage *img5 = cvLoadImage(im5, CV_LOAD_IMAGE_UNCHANGED);

  cvShowImage("1", img1);
  cvShowImage("2", img2);
  cvShowImage("3", img3);
  cvShowImage("4", img4);
  cvShowImage("5", img5);
  //cvNamedWindow( "Image", 1 );
  //cvShowManyImages("Image", 5, img1, img2, img3, img4, img5);
  //cvShowManyImages("ImageMatches", 2, img1, img2);

  //cvShowManyImages("Image", 1, img1);

  std::cout << "Max Match Score: " << maxMatchScore << std::endl;

  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << "Usage: ./2dMatcher" << std::endl; }
