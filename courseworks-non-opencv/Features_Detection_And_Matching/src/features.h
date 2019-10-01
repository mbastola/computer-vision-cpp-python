#ifndef FEATURES_H
#define FEATURES_H

#include "ImageLib/ImageLib.h"
#include "ImageDatabase.h"

class Fl_Image;

//5x5 Gaussian
const double gaussian5x5[25] = {0.003663, 0.014652,  0.025641,  0.014652,  0.003663, 
                                0.014652, 0.0586081, 0.0952381, 0.0586081, 0.014652, 
                                0.025641, 0.0952381, 0.150183,  0.0952381, 0.025641, 
                                0.014652, 0.0586081, 0.0952381, 0.0586081, 0.014652, 
                                0.003663, 0.014652,  0.025641,  0.014652,  0.003663 };



//7x7 Gaussian
const double gaussian7x7[49] = {0.000896861, 0.003587444, 0.006278027, 0.00896861,  0.006278027, 0.003587444, 0.000896861,
                                0.003587444, 0.010762332, 0.023318386, 0.029596413, 0.023318386, 0.010762332, 0.003587444, 
                                0.006278027, 0.023318386, 0.049327354, 0.06367713,  0.049327354, 0.023318386, 0.006278027,
                                0.00896861,  0.029596413, 0.06367713,  0.08161435,  0.06367713,  0.029596413, 0.00896861,  
                                0.006278027, 0.023318386, 0.049327354, 0.06367713,  0.049327354, 0.023318386, 0.006278027,
                                0.003587444, 0.010762332, 0.023318386, 0.029596413, 0.023318386, 0.010762332, 0.003587444,
                                0.000896861, 0.003587444, 0.006278027, 0.00896861,  0.006278027, 0.003587444, 0.000896861};

const double gaussian7x7_sigma2_5[49] = {0.008631, 0.012808, 0.016231, 0.017564, 0.016231, 0.012808, 0.008631,
					 0.012808, 0.019007, 0.024086, 0.026064, 0.024086, 0.019007, 0.012808,
					 0.016231, 0.024086, 0.030521, 0.033028, 0.030521, 0.024086, 0.016231,
					 0.017564, 0.026064, 0.033028, 0.035741, 0.033028, 0.026064, 0.017564,
					 0.016231, 0.024086, 0.030521, 0.033028, 0.030521, 0.024086, 0.016231, 
					 0.012808, 0.019007, 0.024086, 0.026064, 0.024086, 0.019007, 0.012808,
					 0.008631, 0.012808, 0.016231, 0.017564, 0.016231, 0.012808, 0.008631};

const double gaussian9x9[81] = { 0, 0.000001, 0.000014, 0.000055, 0.000088, 0.000055, 0.000014, 0.0000010,
				 0.000001, 0.000036, 0.000362, 0.001445, 0.002289, 0.001445, 0.000362, 0.000036, 0.000001,
				 0.000014, 0.000362, 0.003672, 0.014648, 0.023205, 0.014648, 0.003672, 0.000362, 0.000014,
				 0.000055, 0.001445, 0.014648, 0.058434, 0.092566, 0.058434, 0.014648, 0.001445, 0.000055,
				 0.000088, 0.002289, 0.023205, 0.092566, 0.146634, 0.092566, 0.023205, 0.002289, 0.000088,
				 0.000055, 0.001445, 0.014648, 0.058434, 0.092566, 0.058434, 0.014648, 0.001445, 0.000055,
				 0.000014, 0.000362, 0.003672, 0.014648, 0.023205, 0.014648, 0.003672, 0.000362, 0.000014,
				 0.000001, 0.000036, 0.000362, 0.001445, 0.002289, 0.001445, 0.000362, 0.000036, 0.000001,
				 0, 0.000001, 0.000014, 0.000055, 0.000088, 0.000055, 0.000014, 0.000001, 0 };
  
struct ROCPoint
{
	double trueRate;
	double falseRate;
};


// Compute harris values of an image.
void computeHarrisValues(CFloatImage &srcImage, CFloatImage &harrisImage);

  
void computeHarrisValuesWithThetas(CFloatImage &srcImage, CFloatImage &harrisImage, CFloatImage& harrisThetaImage);

//  Compute local maximum of Harris values in an image.
void computeLocalMaxima(CFloatImage &srcImage,CByteImage &destImage);

// Compute features of an image.
bool computeFeatures(CFloatImage &image, FeatureSet &features, int featureType);

// Perform a query on the database.
bool performQuery(const FeatureSet &f1, const ImageDatabase &db, int &bestIndex, vector<FeatureMatch> &bestMatches, double &bestScore, int matchType);

// Match one feature set with another.
bool matchFeatures(const FeatureSet &f, const FeatureSet &f2, vector<FeatureMatch> &matches, double &totalScore, int matchType);

// Add ROC curve data to the data vector
void addRocData(const FeatureSet &f1, const FeatureSet &f2, const vector<FeatureMatch> &matches, double h[9],vector<bool> &isMatch,double threshold,double &maxD);

// Evaluate a match using a ground truth homography.
double evaluateMatch(const FeatureSet &f1, const FeatureSet &f2, const vector<FeatureMatch> &matches, double h[9]);

// Compute silly example features.
void dummyComputeFeatures(CFloatImage &image, FeatureSet &features);
//Illumination & rotation invariant 20x20 window size 16 cells feature detector
void manilsfeatureDescriptor(CFloatImage& image, Feature& feature);

void simple5x5featureDescriptor(CFloatImage& image, Feature& feature);

// Compute actual feature with simple 5x5 descriptor
void ComputeHarrisFeatures(CFloatImage &image, FeatureSet &features);

// Compute actual feature with simple manils 20x20 descriptor
void ComputeHarrisFeaturesAdvanced(CFloatImage &image, FeatureSet &features);

// Perform ssd feature matching.
void ssdMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, double &totalScore);

// Perform ratio feature matching.  You must implement this.
void ratioMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, double &totalScore);

// Convert Fl_Image to CFloatImage.
bool convertImage(const Fl_Image *image, CFloatImage &convertedImage);

// Convert CFloatImage to CByteImage.
void convertToByteImage(CFloatImage &floatImage, CByteImage &byteImage);

// Compute SSD distance between two vectors.
double distanceSSD(const vector<double> &v1, const vector<double> &v2);

// Transform point by homography.
void applyHomography(double x, double y, double &xNew, double &yNew, double h[9]);

// Computes points on the Roc curve
vector<ROCPoint> computeRocCurve(vector<FeatureMatch> &matches,vector<bool> &isMatch,vector<double> &thresholds);

// Compute AUC given a ROC curve
double computeAUC(vector<ROCPoint> &results);

#endif
