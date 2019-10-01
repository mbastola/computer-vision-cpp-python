#include <assert.h>
#include <math.h>
#include <FL/Fl.H>
#include <FL/Fl_Image.H>
#include "features.h"
#include "ImageLib/FileIO.h"
#include <iostream>
#define PI 3.14159265358979323846

double HARRIS_THRES = 0.0001;
double GRADIENT_THRES =  0.0000001;
  
// Compute features of an image.
bool computeFeatures(CFloatImage &image, FeatureSet &features, int featureType) {
	// TODO: Instead of calling dummyComputeFeatures, write your own
	// feature computation routines and call them here.
	switch (featureType) {
	case 1:
	  dummyComputeFeatures(image, features);
	  break;
	case 2:
	  /*SImple 5 x5 descriptor*/
	  ComputeHarrisFeatures(image, features);
	  break;
	case 3:
	  /*Manils 20x20 descriptor that is Rotation and illumination invariant*/
	  ComputeHarrisFeaturesAdvanced(image, features);
	  break;
	default:
	  return false;
	}

	// This is just to make sure the IDs are assigned in order, because
	// the ID gets used to index into the feature array.
	for (unsigned int i=0; i<features.size(); i++) {
		features[i].id = i+1;
	}

	return true;
}

// Perform a query on the database.  This simply runs matchFeatures on
// each image in the database, and returns the feature set of the best
// matching image.
bool performQuery(const FeatureSet &f, const ImageDatabase &db, int &bestIndex, vector<FeatureMatch> &bestMatches, double &bestScore, int matchType) {
	// Here's a nice low number.
	bestScore = -1e100;

	vector<FeatureMatch> tempMatches;
	double tempScore;

	for (unsigned int i=0; i<db.size(); i++) {
		if (!matchFeatures(f, db[i].features, tempMatches, tempScore, matchType)) {
			return false;
		}

		if (tempScore > bestScore) {
			bestIndex = i;
			bestScore = tempScore;
			bestMatches = tempMatches;
		}
	}

	return true;
}

// Match one feature set with another.
bool matchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, double &totalScore, int matchType) {
	// TODO: We have given you the ssd matching function, you must write your own
	// feature matching function for the ratio test.
	
	printf("\nMatching features.......\n");

	switch (matchType) {
	case 1:
		ssdMatchFeatures(f1, f2, matches, totalScore);
		return true;
	case 2:
		ratioMatchFeatures(f1, f2, matches, totalScore);
		return true;
	default:
		return false;
	}
}

// Evaluate a match using a ground truth homography.  This computes the
// average SSD distance between the matched feature points and
// the actual transformed positions.
double evaluateMatch(const FeatureSet &f1, const FeatureSet &f2, const vector<FeatureMatch> &matches, double h[9]) {
	double d = 0;
	int n = 0;

	double xNew;
	double yNew;

    unsigned int num_matches = matches.size();
	for (unsigned int i=0; i<num_matches; i++) {
		int id1 = matches[i].id1;
        int id2 = matches[i].id2;
        applyHomography(f1[id1-1].x, f1[id1-1].y, xNew, yNew, h);
		d += sqrt(pow(xNew-f2[id2-1].x,2)+pow(yNew-f2[id2-1].y,2));
		n++;
	}	

	return d / n;
}

void addRocData(const FeatureSet &f1, const FeatureSet &f2, const vector<FeatureMatch> &matches, double h[9],vector<bool> &isMatch,double threshold,double &maxD) {
	double d = 0;

	double xNew;
	double yNew;

    unsigned int num_matches = matches.size();
	for (unsigned int i=0; i<num_matches; i++) {
		int id1 = matches[i].id1;
        int id2 = matches[i].id2;
		applyHomography(f1[id1-1].x, f1[id1-1].y, xNew, yNew, h);

		// Ignore unmatched points.  There might be a better way to
		// handle this.
		d = sqrt(pow(xNew-f2[id2-1].x,2)+pow(yNew-f2[id2-1].y,2));
		if (d<=threshold)
		{
			isMatch.push_back(1);
		}
		else
		{
			isMatch.push_back(0);
		}

		if (matches[i].score>maxD)
			maxD=matches[i].score;
	}	
}

vector<ROCPoint> computeRocCurve(vector<FeatureMatch> &matches,vector<bool> &isMatch,vector<double> &thresholds)
{
	vector<ROCPoint> dataPoints;

	for (int i=0; i < (int)thresholds.size();i++)
	{
		//printf("Checking threshold: %lf.\r\n",thresholds[i]);
		int tp=0;
		int actualCorrect=0;
		int fp=0;
		int actualError=0;
		int total=0;

        int num_matches = (int) matches.size();
		for (int j=0;j < num_matches;j++)
		{
			if (isMatch[j])
			{
				actualCorrect++;
				if (matches[j].score<thresholds[i])
				{
					tp++;
				}
			}
			else
			{
				actualError++;
				if (matches[j].score<thresholds[i])
				{
					fp++;
				}
            }
			
			total++;
		}

		ROCPoint newPoint;
		//printf("newPoints: %lf,%lf",newPoint.trueRate,newPoint.falseRate);
		if (actualCorrect != 0){
		  newPoint.trueRate=(double(tp)/actualCorrect);
		}
		else{
		  newPoint.trueRate=0;
		}
		if (actualError != 0){
		  newPoint.falseRate=(double(fp)/actualError);
		}
		else{
		  newPoint.falseRate=0;
		}
		//printf("newPoints: %lf,%lf",newPoint.trueRate,newPoint.falseRate);
		dataPoints.push_back(newPoint);
	}
	return dataPoints;
}


// Compute silly example features.  This doesn't do anything
// meaningful.
void dummyComputeFeatures(CFloatImage &image, FeatureSet &features) {
	CShape sh = image.Shape();
	Feature f;

	for (int y=0; y<sh.height; y++) {
		for (int x=0; x<sh.width; x++) {
			double r = image.Pixel(x,y,0);
			double g = image.Pixel(x,y,1);
			double b = image.Pixel(x,y,2);

			if ((int)(255*(r+g+b)+0.5) % 100  == 1) {
				// If the pixel satisfies this meaningless criterion,
				// make it a feature.
				
				f.type = 1;
				f.id += 1;
				f.x = x;
				f.y = y;

				f.data.resize(1);
				f.data[0] = r + g + b;

				features.push_back(f);
			}
		}
	}
}

void ComputeHarrisFeatures(CFloatImage &image, FeatureSet &features)
{
  //Create grayscale image used for Harris detection
  CFloatImage grayImage=ConvertToGray(image);
  
  //Create image to store Harris values
  CFloatImage harrisImage(image.Shape().width,image.Shape().height,1);
  
  //Create image to store local maximum harris values as 1, other pixels 0
  CByteImage harrisMaxImage(image.Shape().width,image.Shape().height,1);
    
  //compute Harris values puts harris values at each pixel position in harrisImage. 
  //You'll need to implement this function.
  computeHarrisValues(grayImage, harrisImage);

  // Threshold the harris image and compute local maxima.  You'll need to implement this function.
  computeLocalMaxima(harrisImage,harrisMaxImage);
  
  // Prints out the harris image for debugging purposes
  CByteImage tmp(harrisImage.Shape());
  convertToByteImage(harrisImage, tmp);
  WriteFile(tmp, "harris.tga");
    

  // TO DO--------------------------------------------------------------------
  //Loop through feature points in harrisMaxImage and create feature descriptor 
  //for each point above a threshold
  
    for (int y=0;y<harrisMaxImage.Shape().height;y++) {
      for (int x=0;x<harrisMaxImage.Shape().width;x++) {
	
	// Skip over non-maxima
	if (harrisMaxImage.Pixel(x, y, 0) == 0)
	  continue;
	
	//TO DO---------------------------------------------------------------------
	 // Fill in feature with descriptor data here. 
	Feature f;
	f.type = 1; //temporary
	f.id += 1;
	f.x = x;
	f.y = y;
	
	simple5x5featureDescriptor(grayImage, f);
	// Add the feature to the list of features
	features.push_back(f); 
      }
    }
}


void ComputeHarrisFeaturesAdvanced(CFloatImage &image, FeatureSet &features)
{
  //Create grayscale image used for Harris detection
  CFloatImage grayImage=ConvertToGray(image);
  
  //Create image to store Harris values
  CFloatImage harrisImage(image.Shape().width,image.Shape().height,1);
  
  //Create image to store local maximum harris values as 1, other pixels 0
  CByteImage harrisMaxImage(image.Shape().width,image.Shape().height,1);
  CFloatImage harrisThetaImage(image.Shape().width,image.Shape().height,1);
  
  //compute Harris values puts harris values at each pixel position in harrisImage. 
  //You'll need to implement this function.
  
  computeHarrisValuesWithThetas(grayImage, harrisImage, harrisThetaImage);
    
  // Threshold the harris image and compute local maxima.  You'll need to implement this function.
  computeLocalMaxima(harrisImage,harrisMaxImage);
  
  // Prints out the harris image for debugging purposes
  CByteImage tmp(harrisImage.Shape());
  convertToByteImage(harrisImage, tmp);
  WriteFile(tmp, "harris.tga");
    

  // TO DO--------------------------------------------------------------------
  //Loop through feature points in harrisMaxImage and create feature descriptor 
  //for each point above a threshold
  
    for (int y=0;y<harrisMaxImage.Shape().height;y++) {
      for (int x=0;x<harrisMaxImage.Shape().width;x++) {
	
	// Skip over non-maxima
	if (harrisMaxImage.Pixel(x, y, 0) == 0)
	  continue;
	
	//TO DO---------------------------------------------------------------------
	 // Fill in feature with descriptor data here. 
	Feature f;
	f.type = 1; //temporary
	f.id += 1;
	f.x = x;
	f.y = y;
	f.angleRadians = harrisThetaImage.Pixel(x, y, 0);
	
	manilsfeatureDescriptor(grayImage, f);
	// Add the feature to the list of features

	features.push_back(f); 
      }
    }
}


/**Takes 5x5 square window around detected feature
   Compute edge orientation (angle of the gradient - 90°) for each pixel
   Throws out weak edges (threshold gradient magnitude)
   Creates histogram of surviving edge orientations
*/
void simple5x5featureDescriptor(CFloatImage& image, Feature& feature){
  feature.data.resize(8); //for 1 5x5 cell with 8 orientations (l,b,r,t,bl,tr,br,tl) implementation
  
  int blockSize = 5; //window size
  int w = image.Shape().width;
  int h = image.Shape().height;
  
  int offsetx = feature.x - blockSize/2;
  int offsety = feature.y - blockSize/2;
  
  for (int j=0; j < blockSize; j++)
    {
      for (int i=0; i < blockSize ; i++)
	{
	  int x = offsetx + i; //integer division to get start offset
	  int y = offsety + j;
	  
	  double gradientMag[8] = {0,0,0,0,0,0,0,0}; //l,b,r,t,bl,tr,br,tl orientations
	  //int[] gradientAngle = [180,-90,0,90,-135,45,-45,135];
	  //int edgeAngle[8] = [-45,-180,-90,0, 135,-45,-135,45];
	  if (x-1 >= 0){
	    gradientMag[0] = image.Pixel(x-1, y, 0)-image.Pixel(x, y, 0);
	  }
	  if (y-1 >= 0){
	    gradientMag[1] = image.Pixel(x, y-1, 0)-image.Pixel(x, y, 0);
	  }
	  if (x+1 < w){
	    gradientMag[2] = image.Pixel(x+1, y, 0)-image.Pixel(x, y, 0);
	  }
	  if (y+1 < h){
	    gradientMag[3] = image.Pixel(x, y+1, 0)-image.Pixel(x, y, 0);
      }
	  if (x-1 >= 0 && y-1 >= 0 ){
	    gradientMag[4] = image.Pixel(x-1, y-1, 0)-image.Pixel(x, y, 0);
	  }
	  if (x+1 < w && y+1 < h ){
	    gradientMag[5] = image.Pixel(x+1, y+1, 0)-image.Pixel(x, y, 0);
	  }
	  if (x+1 < w && y-1 >= 0 ){
	    gradientMag[6] = image.Pixel(x+1, y-1, 0)-image.Pixel(x, y, 0);
	  }
	  if (x-1 >= 0 && y+1 < h ){
	    gradientMag[7] = image.Pixel(x-1, y+1, 0)-image.Pixel(x, y, 0);
	  }
	  
	  int maxPos = 0;
	  double maxVal = -1e100;
	  for (int k=0; k < 8; ++k){
	    if (maxVal < gradientMag[k]){
	      maxVal = gradientMag[k];
	      maxPos = k;
	    }
	  }
	  if (maxVal > GRADIENT_THRES){
	    feature.data[maxPos]++ ; //create histogram of angles
	  }
	}
    }
}

void manilsfeatureDescriptor(CFloatImage& image, Feature& feature)
{
  feature.data.resize(128); //for 16 cells with 8 orientations (l,b,r,t,bl,tr,br,tl) implementation
  
  int winSize = 20; //window size 20x20
  int cellSize = 5;
  int w = image.Shape().width;
  int h = image.Shape().height;

  int offsetx = feature.x - winSize/2; //integer division to get start offset
  int offsety = feature.y - winSize/2;
  
  double angle = 0;
  if (!isnan(feature.angleRadians) && !(feature.angleRadians > 2*PI || feature.angleRadians < -2*PI )){
    angle = feature.angleRadians * 180/PI;
  }
 
  int angleOffset = (int)angle / 45;
		 
  int cell_counter = 0;
  for (int j=0; j < winSize; j+= cellSize)
    {
      for (int i=0; i < winSize ; i+= cellSize)
	{
	  double meanVal = 0.0;
	  double stdDev = 0.0;
	  for (int cell_j=0; cell_j < cellSize; cell_j++)
	    {
	      for (int cell_i=0; cell_i < cellSize ; cell_i++)
		{
		  int x = offsetx + i*cellSize + cell_i; 
		  int y = offsety + j*cellSize + cell_j;
		  meanVal += image.Pixel(x, y, 0);
		}
	    }
	  meanVal /= pow(cellSize,2);
	  for (int cell_j=0; cell_j < cellSize; cell_j++)
	    {
	      for (int cell_i=0; cell_i < cellSize ; cell_i++)
		{
		  int x = offsetx + i*cellSize + cell_i; 
		  int y = offsety + j*cellSize + cell_j;
		  stdDev += pow((image.Pixel(x, y, 0)-meanVal),2);
		}
	    }
	  stdDev /= pow(cellSize,2);
	  stdDev = sqrt(stdDev);
 
	  for (int cell_j=0; cell_j < cellSize; cell_j++)
	    {
	      for (int cell_i=0; cell_i < cellSize ; cell_i++)
		{
		  int x = offsetx + i*cellSize + cell_i; 
		  int y = offsety + j*cellSize + cell_j;
		  double gradientMag[8] = {0,0,0,0,0,0,0,0}; 
		  //int[] gradientAngle = [-135,-90,-45,0,45,90,135,180];		  
		  if (x-1 >= 0 && y-1 >= 0 ){
		    gradientMag[(0+angleOffset)%8] = (image.Pixel(x-1, y-1, 0)-image.Pixel(x, y, 0))/stdDev*gaussian7x7_sigma2_5[7*(cell_j+1)+(cell_i+1)];
		  }
		  if (y-1 >= 0){
		    gradientMag[(1+angleOffset)%8] = (image.Pixel(x, y-1, 0)-image.Pixel(x, y, 0))/stdDev*gaussian7x7_sigma2_5[7*(cell_j+1)+(cell_i+1)];
		  }
		  if (x+1 < w && y-1 >= 0 ){
		    gradientMag[(2+angleOffset)%8] = (image.Pixel(x+1, y-1, 0)-image.Pixel(x, y, 0))/stdDev*gaussian7x7_sigma2_5[7*(cell_j+1)+(cell_i+1)];
		  }
		  if (x+1 < w){
		    gradientMag[(3+angleOffset)%8] = (image.Pixel(x+1, y, 0)-image.Pixel(x, y, 0))/stdDev*gaussian7x7_sigma2_5[7*(cell_j+1)+(cell_i+1)];
		  }
		  if (x+1 < w && y+1 < h ){
		    gradientMag[(4+angleOffset)%8] = (image.Pixel(x+1, y+1, 0)-image.Pixel(x, y, 0))/stdDev*gaussian7x7_sigma2_5[7*(cell_j+1)+(cell_i+1)];
		  }
		  if (y+1 < h){
		    gradientMag[(5+angleOffset)%8] = (image.Pixel(x, y+1, 0)-image.Pixel(x, y, 0))/stdDev*gaussian7x7_sigma2_5[7*(cell_j+1)+(cell_i+1)];
		  }
		  if (x-1 >= 0 && y+1 < h ){
		    gradientMag[(6+angleOffset)%8] = (image.Pixel(x-1, y+1, 0)-image.Pixel(x, y, 0))/stdDev*gaussian7x7_sigma2_5[7*(cell_j+1)+(cell_i+1)];
		  }
		  if (x-1 >= 0){
		    gradientMag[(7+angleOffset)%8] = (image.Pixel(x-1, y, 0)-image.Pixel(x, y, 0))/stdDev*gaussian7x7_sigma2_5[7*(cell_j+1)+(cell_i+1)];
		  }
		  
		  int maxPos = 0;
		  double maxVal = -1e100;
		  for (int k=0; k < 8; ++k){
		    if (maxVal < gradientMag[k]){
		      maxVal = gradientMag[k];
		      maxPos = k;
		    }
		  }
		  if (maxVal > GRADIENT_THRES){
		    feature.data[cell_counter*8 + maxPos]++ ; //create histogram of angles
		  }
		}
	    }
	  cell_counter++;
	}
    }
}


//TO DO---------------------------------------------------------------------
//Loop through the image to compute the harris corner values as described in class
// srcImage:  grayscale of original image
// harrisImage:  populate the harris values per pixel in this image
void computeHarrisValues(CFloatImage &srcImage, CFloatImage &harrisImage)
{
  int w = srcImage.Shape().width;
  int h = srcImage.Shape().height;
  int blockSize = 5; //window size
  double maxHarrisVal = 0.0;
  double minHarrisVal = 1e100;
  double maxGradVal = 0.0;
  for (int y = 0; y < h-2; y++) {
    for (int x = 0; x < w-2; x++) {

      // TODO:  Compute the harris score for 'srcImage' at this pixel and store in 'harrisImage'.  See the project
      //   page for pointers on how to do this
      //store dI/dx and dI/dy in harrisImage
      
      double sumH11 = 0;
      double sumH22 = 0;
      double sumH12 = 0; //h12 = h21 so only 3 values needed
         
      for (int j = 0; j < blockSize; j++)
	{
	  for (int i = 0; i < blockSize; i++)
	    {
	      if ( (x+i < w-1) && (y+j < h-1) )
		{ 
		  double Ix = srcImage.Pixel(x+i+1, y+j, 0) -  srcImage.Pixel(x+i, y+j, 0);
		  double Iy = srcImage.Pixel(x+i, y+j+1, 0) -  srcImage.Pixel(x+i, y+j, 0);
		  Ix = Ix*gaussian5x5[j*blockSize+i];
		  Iy = Iy*gaussian5x5[j*blockSize+i]; //apply gaussian 5x5 filter
		  if (Ix > maxGradVal || Iy > maxGradVal){
		    if (Ix > Iy)
		      maxGradVal = Ix;
		    else
		      maxGradVal = Iy;
		  }
		  sumH11 += pow(Ix,2);
		  sumH22 += pow(Iy,2);
		  sumH12 += Ix*Iy;
		}
	    }
	}
      
      double C = (sumH11*sumH22 - sumH12*sumH12)/(sumH11 + sumH22);
      harrisImage.Pixel(x+2, y+2, 0) = C;

      if (C > maxHarrisVal){
	maxHarrisVal = C;
      }
      if (C < minHarrisVal){
	minHarrisVal = C;
      }
    }
  }
  HARRIS_THRES = (maxHarrisVal-minHarrisVal)*.06+minHarrisVal;
  GRADIENT_THRES = maxGradVal*.1;
}


//Loop through the image to compute the harris corner values as described in class
// srcImage:  grayscale of original image
// harrisImage:  populate the harris values per pixel in this image
void computeHarrisValuesWithThetas(CFloatImage &srcImage, CFloatImage &harrisImage, CFloatImage &harrisThetaImage)
{
  int w = srcImage.Shape().width;
  int h = srcImage.Shape().height;
  int blockSize = 5; //window size
  double maxHarrisVal = 0.0;
  double minHarrisVal = 1e100;
  double maxGradVal = 0.0;
  for (int y = 0; y < h-2; y++) {
    for (int x = 0; x < w-2; x++) {

      // TODO:  Compute the harris score for 'srcImage' at this pixel and store in 'harrisImage'.  See the project
      //   page for pointers on how to do this
      //store dI/dx and dI/dy in harrisImage
      
      double sumH11 = 0;
      double sumH22 = 0;
      double sumH12 = 0; //h12 = h21 so only 3 values needed
         
      for (int j = 0; j < blockSize; j++)
	{
	  for (int i = 0; i < blockSize; i++)
	    {
	      if ( (x+i < w-1) && (y+j < h-1) )
		{ 
		  double Ix = srcImage.Pixel(x+i+1, y+j, 0) -  srcImage.Pixel(x+i, y+j, 0);
		  double Iy = srcImage.Pixel(x+i, y+j+1, 0) -  srcImage.Pixel(x+i, y+j, 0);
		  Ix = Ix*gaussian5x5[j*blockSize+i];
		  Iy = Iy*gaussian5x5[j*blockSize+i]; //apply gaussian 5x5 filter
		  if (Ix > maxGradVal || Iy > maxGradVal){
		    if (Ix > Iy)
		      maxGradVal = Ix;
		    else
		      maxGradVal = Iy;
		  }
		  sumH11 += pow(Ix,2);
		  sumH22 += pow(Iy,2);
		  sumH12 += Ix*Iy;
		}
	    }
	}

      double det = sumH11*sumH22 - sumH12*sumH12;
      double trace = sumH11 + sumH22;
      double C = det/trace;
      double eig1 = trace/2 + sqrt(pow(trace,2)/4 - det); //source: http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
      double theta = atan2(sumH12, eig1 - sumH22);
      harrisImage.Pixel(x+2, y+2, 0) = C;
      harrisThetaImage.Pixel(x+2, y+2, 0) = theta;
      
      if (C > maxHarrisVal){
	maxHarrisVal = C;
      }
      if (C < minHarrisVal){
	minHarrisVal = C;
      }
    }
  }
  HARRIS_THRES = (maxHarrisVal-minHarrisVal)*.06+minHarrisVal;
  GRADIENT_THRES = maxGradVal*.1;
}



// TO DO---------------------------------------------------------------------
// Loop through the harrisImage to threshold and compute the local maxima in a neighborhood
// srcImage:  image with Harris values
// destImage: Assign 1 to a pixel if it is above a threshold and is the local maximum in 3x3 window, 0 otherwise.
//    You'll need to find a good threshold to use.
void computeLocalMaxima(CFloatImage &srcImage,CByteImage &destImage)
{
  int w = srcImage.Shape().width;
  int h = srcImage.Shape().height;
  int blockSize = 3; //window size
  
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      double localMaximaVal = 0;
      int maxLocX = 0;
      int maxLocY = 0;
      for (int j = 0; j < blockSize; j++)
	{
	  for (int i = 0; i < blockSize; i++)
	    {
	      if ( (x+i < w) && (y+j < h) )
		{
		  double pixelVal = srcImage.Pixel(x+i, y+j, 0);
		  if (pixelVal > localMaximaVal){
		    localMaximaVal = pixelVal;
		    maxLocX = x+i;
		    maxLocY = y+j;
		  }
		}
	    }
	}
	  
      if (localMaximaVal > HARRIS_THRES){
	destImage.Pixel(maxLocX, maxLocY, 0) = 1;
      }
    }
    
  }
  //CByteImage tmp(destImage.Shape());
  //convertToByteImage(destImage, tmp);
  WriteFile(destImage, "bitmap.tga");
}

// Perform simple feature matching.  This just uses the SSD
// distance between two feature vectors, and matches a feature in the
// first image with the closest feature in the second image.  It can
// match multiple features in the first image to the same feature in
// the second image.
void ssdMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, double &totalScore) {
	int m = f1.size();
	int n = f2.size();

	matches.resize(m);
	totalScore = 0;

	double d;
	double dBest;
	int idBest;

	for (int i=0; i<m; i++) {
		dBest = 1e100;
		idBest = 0;

		for (int j=0; j<n; j++) {
			d = distanceSSD(f1[i].data, f2[j].data);

			if (d < dBest) {
				dBest = d;
				idBest = f2[j].id;
			}
		}
		
        matches[i].id1 = f1[i].id;
	matches[i].id2 = idBest;
	matches[i].score = dBest;
	totalScore += matches[i].score;
	}
}

// Compute SSD distance between two vectors.
double distanceSSD(const vector<double> &v1, const vector<double> &v2) {
	int m = v1.size();
	int n = v2.size();

	if (m != n) {
		// Here's a big number.
		return 1e100;
	}

	double dist = 0;

	for (int i=0; i<m; i++) {
		dist += pow(v1[i]-v2[i], 2);
	}

	return sqrt(dist);
}

// TODO: Write this function to perform ratio feature matching.  
// This just uses the ratio of the SSD distance of the two best matches as the score
// and matches a feature in the first image with the closest feature in the second image.
// It can match multiple features in the first image to the same feature in
// the second image.  (See class notes for more information, and the sshMatchFeatures function above as a reference)
void ratioMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, double &totalScore) 
{
  int m = f1.size();
  int n = f2.size();
  
  matches.resize(m);
  totalScore = 0;
  
  double d;
  double dBest;
  int idBest;
  double dSecondBest;
  int idSecondBest;
  
  for (int i=0; i<m; i++) {
    dBest = 1e100;
    idBest = 0;
    dSecondBest = 1e100;
    idSecondBest = 0;
    
    for (int j=0; j<n; j++) {
      d = distanceSSD(f1[i].data, f2[j].data);
      
      if (d < dBest) {
	dSecondBest = dBest;
	idSecondBest = idBest; //previous min becomes 2nd most min
	
	dBest = d;
	idBest = f2[j].id;
      }
    }
    
    matches[i].id1 = f1[i].id;
    matches[i].id2 = idBest;
    if (dSecondBest != 0){
      matches[i].score = dBest/dSecondBest;
    }
    else{
      matches[i].score = 1;
    }
    totalScore += matches[i].score;
  }	
}


// Convert Fl_Image to CFloatImage.
bool convertImage(const Fl_Image *image, CFloatImage &convertedImage) {
	if (image == NULL) {
		return false;
	}

	// Let's not handle indexed color images.
	if (image->count() != 1) {
		return false;
	}

	int w = image->w();
	int h = image->h();
	int d = image->d();

	// Get the image data.
	const char *const *data = image->data();

	int index = 0;

	for (int y=0; y<h; y++) {
		for (int x=0; x<w; x++) {
			if (d < 3) {
				// If there are fewer than 3 channels, just use the
				// first one for all colors.
				convertedImage.Pixel(x,y,0) = ((uchar) data[0][index]) / 255.0f;
				convertedImage.Pixel(x,y,1) = ((uchar) data[0][index]) / 255.0f;
				convertedImage.Pixel(x,y,2) = ((uchar) data[0][index]) / 255.0f;
			}
			else {
				// Otherwise, use the first 3.
				convertedImage.Pixel(x,y,0) = ((uchar) data[0][index]) / 255.0f;
				convertedImage.Pixel(x,y,1) = ((uchar) data[0][index+1]) / 255.0f;
				convertedImage.Pixel(x,y,2) = ((uchar) data[0][index+2]) / 255.0f;
			}

			index += d;
		}
	}
	
	return true;
}

// Convert CFloatImage to CByteImage.
void convertToByteImage(CFloatImage &floatImage, CByteImage &byteImage) {
	CShape sh = floatImage.Shape();

    assert(floatImage.Shape().nBands == byteImage.Shape().nBands);
	for (int y=0; y<sh.height; y++) {
		for (int x=0; x<sh.width; x++) {
			for (int c=0; c<sh.nBands; c++) {
				float value = floor(255*floatImage.Pixel(x,y,c) + 0.5f);

				if (value < byteImage.MinVal()) {
					value = byteImage.MinVal();
				}
				else if (value > byteImage.MaxVal()) {
					value = byteImage.MaxVal();
				}

				// We have to flip the image and reverse the color
				// channels to get it to come out right.  How silly!
				byteImage.Pixel(x,sh.height-y-1,sh.nBands-c-1) = (uchar) value;
			}
		}
	}
}


// Transform point by homography.
void applyHomography(double x, double y, double &xNew, double &yNew, double h[9]) {
	double d = h[6]*x + h[7]*y + h[8];

	xNew = (h[0]*x + h[1]*y + h[2]) / d;
	yNew = (h[3]*x + h[4]*y + h[5]) / d;
}

// Compute AUC given a ROC curve
double computeAUC(vector<ROCPoint> &results)
{
	double auc=0;
	double xdiff,ydiff;
	for (int i = 1; i < (int) results.size(); i++)
    {
        //fprintf(stream,"%lf\t%lf\t%lf\n",thresholdList[i],results[i].falseRate,results[i].trueRate);
		xdiff=(results[i].falseRate-results[i-1].falseRate);
		ydiff=(results[i].trueRate-results[i-1].trueRate);
		auc=auc+xdiff*results[i-1].trueRate+xdiff*ydiff/2;
    	    
    }
	return auc;
}
