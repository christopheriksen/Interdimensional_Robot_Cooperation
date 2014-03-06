/**
 * 2dMatcherScoring.cpp
 * Reads in the file storage with the descriptors and compares them to the new image
 * Returns the best match
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

#include "ros/ros.h"
#include "std_msgs/String.h"

using namespace std;
using namespace cv;

void readme();


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

string imageMatcher(string imageName, vector<Mat> descriptors, vector< vector<KeyPoint> > keypoints) { 
//#if 0
  std::vector<double> matchScores;
  std::vector<double> maxMatchScores;
  std::vector<double> pixelAverages;
  std::vector<double> bestPixelScores;
  double maxMatchScore = -1.0;

  // detect and extract features for the new image
  Mat img_1 = imread( imageName, IMREAD_GRAYSCALE );
  int minHessian = 200;
  SurfFeatureDetector detector( minHessian );
  std::vector<KeyPoint> keypoints_1;
  detector.detect( img_1, keypoints_1 );
  SurfDescriptorExtractor extractor;
  Mat descriptors_1;
  extractor.compute( img_1, keypoints_1, descriptors_1 );

  Mat descriptors_2;
  vector<KeyPoint> keypoints_2;

  // loop through each image pair doing the comparisons
  for (size_t j=0; j != descriptors.size(); ++j) {
    descriptors_2 = descriptors[j];
    keypoints_2 = keypoints[j]; // we should have the same number of keypoint and descriptor vectors

    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors[j], matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    { 
      double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    { 
      if( matches[i].distance < 2*min_dist )
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
        }
    }

    //-- Draw only "good" matches
    Mat img_matches;

    //drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                 // good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 // vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

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
    }

    double totalPixels = 0.0;
    for (size_t i = 0; i < pixelMatchScores.size(); ++i) {
      totalPixels += 10000.0/pixelMatchScores[i];
    }

    if (pixelMatchScores.size() == 0) {
      pixelAverages.push_back(0);
      bestPixelScores.push_back(0);
    } else {
      pixelAverages.push_back(totalPixels);
      bestPixelScores.push_back(totalPixels);
    }

    cout << matchScore << endl;

    for (size_t i = 0; i < inFrame.size(); ++i) {
      double dist = (double)(inFrame[i].distance);
      double pointScore = 1.0/dist;
      matchScore += pointScore;
      std::cout << "distance: " << dist << std::endl;
      std::cout << "pointScore: " << pointScore << std::endl;
    }
    std::cout << "matchScore for " << j << ": " << matchScore << std::endl;

    if (matchScore > maxMatchScore) {
      maxMatchScore = matchScore;
    //  maxMatchScores.insert(maxMatchScores.begin(),maxMatchScore);
    }

    matchScores.push_back(matchScore);
    maxMatchScores.push_back(matchScore);

    ///////////////// SAVE COMPARISONS /////////////////

    // Write the image
    // vector<int> compression_params;
    // compression_params.push_back(3);
    // std::string writeString = fileNames[3]; //"/home/robotics/Desktop/FLANNmatches/matches";
    // writeString += countString;
    // writeString += ".png";
    // imwrite(writeString, img_matches, compression_params);
  } // end of for loop to cycle through all matches

  cout << "top5MatchScores" << endl;
  sort(maxMatchScores.begin(), maxMatchScores.end());
  cout << "1" << endl;
  reverse(maxMatchScores.begin(), maxMatchScores.end());
  cout << "2" << endl;
  std::vector<double> top5MatchScores;
  cout << "3" << endl;
  cout << "max match scores size" << maxMatchScores.size() << endl;

  for (size_t i = 0; i < 5; ++i) {
    cout << "4" << endl;
    top5MatchScores.push_back(maxMatchScores[i]);
    cout << "5" << endl;
    cout << top5MatchScores[i] << endl;
    cout << "6" << endl;
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

  sort(bestPixelScores.begin(), bestPixelScores.end());
  reverse(bestPixelScores.begin(), bestPixelScores.end());

  std::vector<double> nonzeroPixelScores;

  // Pixel shifts of 0 are non-matches for the image, so we remove them
  for (size_t i = 0; i < bestPixelScores.size(); ++i) {
    if (bestPixelScores[i] != 0) {
      nonzeroPixelScores.push_back(bestPixelScores[i]);
      cout << i << " " << bestPixelScores[i] << endl;
    }
  }

  bool plusPixelPlaceHolder = false;

  /*int pixelPlaceHolder = 0;
  while (top5Read.size() < 5) {
    for (size_t i = 0; i < pixelAverages.size(); ++i) {
      for (size_t j = 0; j < top5MatchScores.size(); ++j) {
        if (matchScores[i] == top5MatchScores[j]) {
          top5FileRead = fileNames[3];
          if(pixelAverages[i] == nonzeroPixelScores[pixelPlaceHolder]) {
            ++pixelPlaceHolder;
            plusPixelPlaceHolder = true;
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

  std::cout << "Max Match Score: " << maxMatchScore << std::endl;

  cout << "matchScores" << endl;
  for (int i = 0; i < matchScores.size(); ++i) {
    cout << i+1 << ": " << matchScores[i] << endl;
  }

  cout << "pixelAverages" << endl;
  for (int i = 0; i < pixelAverages.size(); ++i) {
    cout << i+1 << ": " << pixelAverages[i] << endl;
  }

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

  std::cout << "Max Match Score: " << maxMatchScore << std::endl;*/
  //#endif
  return "hi";

  //waitKey(0);
  // return the best match
} // end of matcher function

int main( int argc, char** argv ) {
  // read in match information for each image in the directory (from a text file) and then store
  // the info in vectors (loop)
  // currently, the first thing in the text file is the directory of filestorage files
  // and the second line is the image we're trying to match
  ifstream infile;
  ifstream in_stream;
  vector<string> fileNames;
  string line;
  in_stream.open("/home/robotics/Desktop/KristinasDesktopTo01_20_2014/scoringConfig.txt");

  while(!in_stream.eof())
  {
      in_stream >> line;
      fileNames.push_back(line);
  }

  in_stream.close();

  ///////////////////////////// BEGIN DESCRIPTORS /////////////////////////////

  string directory = fileNames[0];
  directory += "*.xml";
  cout << directory << endl;

  const char * fileDirec;
  fileDirec = directory.c_str();

  // Get the number of files in the directory
  glob_t gl;
  size_t numFiles = 0;
  if(glob(fileDirec, GLOB_NOSORT, NULL, &gl) == 0)
    numFiles = gl.gl_pathc;
  globfree(&gl);
  cout << "Number of files: " << numFiles << endl;

  // Get the file names from the directory
  vector<string> descriptNames;
  const char * descDirec;
  descDirec = fileNames[0].c_str();
  string fileExtention = ".xml";
  const char * descExt;
  string extension = "xml";
  descExt = extension.c_str();

  DIR* dirFile = opendir( descDirec );
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
         if ( strstr( hFile->d_name, descExt )) {
          string counts = fileNames[0];
          counts += hFile->d_name;
          descriptNames.push_back(counts);
        }
      } 
      closedir( dirFile );
   }

  sort(descriptNames.begin(), descriptNames.end(), stringCompare);
  for (size_t i = 0; i < descriptNames.size(); ++i) {
    cout << descriptNames[i] << endl;
  }

  int count = 0;

  vector<Mat> descriptors;
  vector< vector<KeyPoint> > keypoints;

  /////////// loop for descriptors ///////////

  for (int k = 0; k < numFiles; ++k) {
    ++count;
    string countString = static_cast<ostringstream*>( &(ostringstream() << count) )->str();

    std::string readString = fileNames[0];
    readString += countString;
    readString += ".xml";

    cout << "pictureNames[" << k << "]: " << descriptNames[k] << endl;

    string descriptorString = "/home/robotics/Desktop/KristinasDesktopTo01_20_2014/mapDescriptors/descriptors";
    descriptorString += countString;
    descriptorString += ".xml";

    string descString = "/home/robotics/Desktop/KristinasDesktopTo01_20_2014/mapDescriptors/desc";
    descString += countString;
    descString += ".xml";


    /////////// read matrix and store info //////////////

    //FileStorage fs;

    Mat matrix;
    FileStorage fs(descriptorString, FileStorage::READ);
    fs["descriptors"] >> matrix;
    descriptors.push_back(matrix);

    /*FileStorage fs(descriptorString, FileStorage::READ);
    FileNode descFileNode = fs["descriptors"];
    read(descFileNode, matrix);
    descriptors.push_back(matrix);*/

    FileStorage fsNew(descString, FileStorage::WRITE);
    write(fsNew, "desc", matrix);
    fsNew.release();
    fs.release();

    cout << "descriptors size: " << descriptors.size() << endl;

    /*fs.open(descriptorString, FileStorage::READ);
    cout << descriptorString << endl;
      //if(fs.isOpened())
      //{
        // Open node
        cv::FileNode d = fs["data"];
        cv::FileNodeIterator it = d.begin(), it_end = d.end();

        for( ; it!= it_end; ++it)
        {
            // Read other data...
            cv::Mat mask;
            (*it)["mask"] >> mask;
            descriptors.push_back(mask);
        }

        //Mat matrix;
        //matrix << fs;
        //descriptors.push_back(fs["featureDescriptors"]);//read(fs["descriptorsSURF"], descriptorsSURF);   
      //}   

      cout << "descriptors size: " << descriptors.size() << endl;

    fs.release();*/
  }

  ////////////////////////////// END DESCRIPTORS //////////////////////////////


  ///////////////////////////// BEGIN KEYPOINTS /////////////////////////////

  string directory2 = fileNames[1];
  directory2 += "*.xml";
  cout << directory2 << endl;

  const char * fileDirec2;
  fileDirec2 = directory2.c_str();

  // Get the number of files in the directory
  glob_t gl2;
  numFiles = 0;
  if(glob(fileDirec2, GLOB_NOSORT, NULL, &gl2) == 0)
    numFiles = gl2.gl_pathc;
  globfree(&gl2);
  cout << "Number of files: " << numFiles << endl;

  // Get the file names from the directory
  vector<string> keypointNames;
  const char * keypointDirec;
  keypointDirec = fileNames[1].c_str();
  string fileExtention2 = ".xml";
  const char * keyExt;
  string extension2 = "xml";
  keyExt= extension2.c_str();

  DIR* dirFile2 = opendir( keypointDirec );
   if ( dirFile2 ) 
   {
      struct dirent* hFile;
      //errno = 0;
      while (( hFile = readdir( dirFile2 )) != NULL ) 
      {
         if ( !strcmp( hFile->d_name, "."  )) continue;
         if ( !strcmp( hFile->d_name, ".." )) continue;

         // dirFile.name is the name of the file. Do whatever string comparison 
         // you want here. Something like:
         if ( strstr( hFile->d_name, keyExt )) {
          string counts = fileNames[1];
          counts += hFile->d_name;
          keypointNames.push_back(counts);
        }
      } 
      closedir( dirFile2 );
   }

  sort(keypointNames.begin(), keypointNames.end(), stringCompare);
  for (size_t i = 0; i < keypointNames.size(); ++i) {
    cout << keypointNames[i] << endl;
  }

  //count = 0;

  /////////// loop for keypoints ///////////

  for (int k = 0; k < numFiles; ++k) {
    //++count;
    count = k+1;
    string countString = static_cast<ostringstream*>( &(ostringstream() << count) )->str();

    std::string readString = fileNames[1];
    readString += countString;
    readString += ".xml";

    cout << "pictureNames[" << k << "]: " << keypointNames[k] << endl;

    string keypointString = "/home/robotics/Desktop/KristinasDesktopTo01_20_2014/mapKeypoints/keypoints";
    keypointString += countString;
    keypointString += ".xml";


    /////////// read matrix and store info //////////////

/*    FileStorage fs;

    fs.open(descriptorString, FileStorage::READ);
      if(fs.isOpened())
      {
        // Open node
        cv::FileNode d = fs["data"];
        cv::FileNodeIterator it = d.begin(), it_end = d.end();

        for( ; it!= it_end; ++it)
        {
            // Read other data...
            cv::Mat mask;
            (*it)["mask"] >> mask;
            descriptors.push_back(mask);
        }   
      }   

    fs.release();*/
//#if 0
    //FileStorage fs2;

    cout << "HERE FIRST" << endl;
    cout << keypointString << endl;

    //fs2.open(keypointString, FileStorage::READ);
      //if(fs2.isOpened())
      //{
        cout << "HERE" << endl;
        #if 0
        // Open node
        cv::FileNode d = fs2["data"];
        cv::FileNodeIterator it = d.begin(), it_end = d.end();

        for( ; it!= it_end; ++it)
        {
            // Read other data...
            vector<KeyPoint> mask;
            (*it)["mask"] >> mask;
            keypoints.push_back(mask);
        }
        #endif


        ///////////////// 2/14/14 FIX??? ///////////////////
        // vector<KeyPoint> keypointVector;
        // keypointVector << fs2;
        // keypoints.push_back(fs2["keypoints"]);//read(fs["descriptorsSURF"], descriptorsSURF);
        
        //////////////// 2/19/14 FIX??? ////////////////////
        vector<KeyPoint> mykpts2;
        FileStorage fs2(keypointString, FileStorage::READ);
        FileNode kptFileNode = fs2["keypoints"];
        read( kptFileNode, mykpts2 );
        cout << "mkpts2 size: " << mykpts2.size() << endl;
        keypoints.push_back(mykpts2);
        // fs2.release();   
        
      //}
      cout << "keypoints size: " << keypoints.size() << endl;

    fs2.release();
   /* for (int i=0; i != keypoints.size(); ++i)
      for (int j=0; j != keypoints[i].size(); ++j)
        cout << keypoints[i][j] << endl;*/
//#endif
  // loop through? 
  /*FileStorage fs2(keypointString, FileStorage::READ);
  FileNode kptFileNode = fs2["keypoints"];
  read( kptFileNode, keypoints );
  fs2.release();*/
  }

  ////////////////////////////// END KEYPOINTS //////////////////////////////
  

  string bestMatchFound;

  bestMatchFound = imageMatcher(fileNames[2], descriptors, keypoints);

  ///////////////////////////////// ROS STARTS HERE /////////////////////////////////////////
#if 0
  ros::init(argc, argv, "listener"); // initialize ROS
  ros::NodeHandle n; // create a ROS node
  ros::Subscriber sub = n.subscribe(/* "Chris's publisher" */, 1000, chatterCallback); // subscribe to image feedback

  string bestMatch = matcher( /*image name*/, descriptors ); // <-- fix image name based on subscriber

  ros::spin();
#endif
  return 0;
}