/**
  * Get the descriptors of each image in a directory of images
  * Store the resulting descriptors in separate file storages to be read in later
 **/

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
#include "opencv2/opencv.hpp"


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
  in_stream.open("/home/robotics/Desktop/KristinasDesktopTo01_20_2014/descriptorsConfig.txt");

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

    string descriptorString = "descriptors";
    descriptorString += countString;
    descriptorString += ".xml";

    string keypointString = "keypoints";
    keypointString += countString;
    keypointString += ".xml";

    FileStorage fs(descriptorString, FileStorage::WRITE);

    //if( !img_1.data || !img_2.data )
    //{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }

    //-- Step 1: Detect the keypoints using SURF Detector
    //int minHessian = 400;
    int minHessian = 200;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> /*keypoints_1,*/ keypoints_2;

    //detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat /*descriptors_1,*/ descriptors_2;

    //extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );

    // store descriptors in a filestorage
    fs << "featureDescriptors" << descriptors_2;

    fs.release();

    cout << keypointString << ": " << keypoints_2.size() << endl;

    FileStorage fStorage(keypointString, FileStorage::WRITE);
    write(fStorage, "keypoints", keypoints_2);
    fStorage.release();

}

  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << "Usage: ./2dMatcher" << std::endl; }
