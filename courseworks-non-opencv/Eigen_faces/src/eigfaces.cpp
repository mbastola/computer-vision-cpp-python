
/////////////////////////////////////////////////////////////////////////////////////////////////
//	Project 4: Eigenfaces                                                                      //
//  CSE 455 Winter 2003                                                                        //
//	Copyright (c) 2003 University of Washington Department of Computer Science and Engineering //
//                                                                                             //
//  File: eigfaces.cpp                                                                         //
//	Author: David Laurence Dewey                                                               //
//	Contact: ddewey@cs.washington.edu                                                          //
//           http://www.cs.washington.edu/homes/ddewey/                                        //
//                                                                                             //
/////////////////////////////////////////////////////////////////////////////////////////////////



#include "stdafx.h"
#include <iterator>

double FACE_MSE_THRES = 600.0;

EigFaces::EigFaces()
:
Faces()
{
	//empty
}

EigFaces::EigFaces(int count, int width, int height)
:
Faces(count, width, height)
{
	//empty
}

void EigFaces::projectFace(const Face& face, Vector& coefficients) const
{
	if (face.getWidth()!=width || face.getHeight()!=height) {
		throw Error("Project: Face to project has different dimensions");
	}

	coefficients.resize(getSize());
	// ----------- TODO #2: compute the coefficients for the face and store in coefficients.
	//(w_i1,...,w_ik)=(u_1'(x_i-mu),...,u_k'(x_i-mu))
	for (int i= 0; i < getSize(); ++i){
	  Face eigFace = (*this)[i]; //u
	  Face avg = this->getAverage(); //mu
	  Face tmp(width, height); 
	  face.sub(avg, tmp);  //x_i-mu
	  coefficients[i] = eigFace.dot(tmp);
	}
}

void EigFaces::constructFace(const Vector& coefficients, Face& result) const
{	
  // ----------- TODO #3: construct a face given the coefficients
  //x=mu+w_1*u_1+w_2*u_2+...+w_n*u_n
  result = this->getAverage();
  for (int i=0;i < getSize(); ++i){
    //Face tmp(result.getWidth(), result.getHeight());
    Face curEigenFace = (*this)[i];//u_i
    curEigenFace *= coefficients[i];//u_i*w_i
    result.add(curEigenFace, result);
  }
}

bool EigFaces::isFace(const Face& face, double max_reconstructed_mse, double& mse) const
{
  // ----------- TODO #4: Determine if an image is a face and return true if it is. Return the actual
  // MSE you calculated for the determination in mse
  // Be sure to test this method out with some face images and some non face images
  // to verify it is working correctly.
  Vector coefficients;
  this->projectFace(face, coefficients);
  Face result;
  this->constructFace(coefficients, result);
  mse = result.mse(face);
  if (mse < max_reconstructed_mse){
    return true;
  }
  return false;  // placeholder, replace
}

bool EigFaces::verifyFace(const Face& face, const Vector& user_coefficients, double max_coefficients_mse, double& mse) const
{
  // ----------- TODO #5 : Determine if face is the same user give the user's coefficients.
  // return the MSE you calculated for the determination in mse.
  Face result;
  this->constructFace(user_coefficients, result);
  mse = result.mse(face);
  if (mse < max_coefficients_mse){
    return true;
  }
  return false;  // placeholder, replace
}

void EigFaces::recognizeFace(const Face& face, Users& users) const
{
  // ----------- TODO #6: Sort the users by closeness of match to the face
  for (int i=0; i < users.getSize(); ++i){
    double user_mse = 0;
    bool dummy = this->verifyFace(face, users[i], 1e100, user_mse);
    users[i].setMse(user_mse);
  }
  users.sort();
}

void EigFaces::findFace(const Image& img, double min_scale, double max_scale, double step, int n, bool crop, Image& result) const
{
  // ----------- TODO #7: Find the faces in Image. Search image scales from min_scale to max_scale inclusive,
  // stepping by step in between. Find the best n faces that do not overlap each other. If crop is true,
  // n is one and you should return the cropped original img in result. The result must be identical
  // to the original besides being cropped. It cannot be scaled and it must be full color. If crop is
  // false, draw green boxes (use r=100, g=255, b=100) around the n faces found. The result must be
  // identical to the original image except for the addition of the boxes.
  std::vector<std::vector<FacePosition> > faces_detected_per_scale;
  for (double scale = min_scale; scale < max_scale; scale+=step)
    {
      std::vector<FacePosition> detected_faces_at_scale;
      this->findFace_at_scale(img, scale, detected_faces_at_scale);
      faces_detected_per_scale.push_back(detected_faces_at_scale);
    }
  //Extract best n faces at all scales based on min error
  std::vector<FacePosition> result_faces;
  for (int j = 0; j < faces_detected_per_scale.size(); ++j){
    for (int i = 0; i < faces_detected_per_scale[j].size(); ++i){
      FacePosition curPos = faces_detected_per_scale[j][i];
      int overlappingWith = 0;
      if( this->isOverlapping(curPos, result_faces, overlappingWith) ){
	if (curPos.error < result_faces[overlappingWith].error){
	  //std::cout << "Swap: "<< result_faces[overlappingWith].x << " " << result_faces[overlappingWith].y << " " << result_faces[overlappingWith].scale << " with " << curPos.x << " " << curPos.y << " " << curPos.scale << std::endl;
	  result_faces[overlappingWith] = curPos;
	}
      }
      else{
	//std::cout << "Add: "<< curPos.x << " " << curPos.y << " " << curPos.scale << std::endl;
	result_faces.push_back(curPos);
      }
    }
  }
  sort(result_faces, 0, result_faces.size());
  while(result_faces.size() > n){
    result_faces.pop_back();
  }
  if(result_faces.size() == 0){
    std::cout << "No face detected :("<< std::endl;
    return; 
  }
  if(crop)
    {
      img.crop(result_faces[0].x,result_faces[0].y,result_faces[0].x+this->width/result_faces[0].scale,result_faces[0].y+this->height/result_faces[0].scale,result);
      return;
    }
  result.resize(img.getWidth(), img.getHeight(), img.getColors());
  img.resample(result);
  for(int i = 0; i < result_faces.size(); ++i){
    //std::cout << "error: " << result_faces[i].error << std::endl;
    //std::cout << "scale: " << result_faces[i].scale << std::endl;
    int start_x = result_faces[i].x;
    int end_x = result_faces[i].x + this->width/result_faces[i].scale;
    int start_y = result_faces[i].y;
    int end_y = result_faces[i].y + this->height/result_faces[i].scale;
    //std::cout << start_x << " " << start_y << " " << end_x << " "<<  end_y << std::endl;
    result.line(start_x, start_y, end_x, start_y, 100, 255, 100);
    result.line(start_x, start_y, start_x, end_y, 100, 255, 100);
    result.line(end_x, start_y, end_x, end_y, 100, 255, 100);
    result.line(start_x, end_y, end_x, end_y, 100, 255, 100);
  }
}

void EigFaces::findFace_at_scale(const Image& img, double scale, std::vector<FacePosition>& detected_faces) const
{
  int new_width = (int)(img.getWidth()*scale);
  int new_height = (int)(img.getHeight()*scale);
  Image scaledImg(new_width, new_height, img.getColors());
  img.resample(scaledImg);
  for (int row = 0; row < scaledImg.getHeight(); ++row){
    for (int col = 0; col < scaledImg.getWidth(); ++col){
      double err = 0;
      Face toFind(this->width, this->height);
      if ( (col+this->width-1 > scaledImg.getWidth() ) || (row+this->height-1 > scaledImg.getHeight() ) ){
	continue;
      }
      scaledImg.crop(col, row, col+this->width-1, row+this->height-1, toFind);
      if (this->isFace(toFind, FACE_MSE_THRES, err)){
	Face tmp;
	toFind.sub(this->getAverage(), tmp);
	double distance = tmp.mag();
	double variance = toFind.var();
	err = err*distance/variance;
	FacePosition fpos;
	fpos.x = col/scale;
	fpos.y = row/scale;
	fpos.scale = scale;
	fpos.error = err;
	int overlappingWith = 0;
	if( this->isOverlapping(fpos, detected_faces, overlappingWith) ){
	  if (fpos.error < detected_faces[overlappingWith].error){
	    detected_faces[overlappingWith] = fpos;
	  }
	}
	else{
	  detected_faces.push_back(fpos);
	}
      }
    }
  } 
}

bool EigFaces::isOverlapping(const FacePosition& pos, const std::vector<FacePosition>& detected_faces, int& overlappingWith) const
{
  int mp_x, mp2_x, mp3_x, mp_y, mp2_y, mp3_y, wd, ht;
  wd = this->width/pos.scale;
  ht = this->height/pos.scale;
  mp_x = pos.x + (int)(wd/2);
  mp_y = pos.y + (int)(ht/2);
  mp2_x = pos.x + (int)(3*wd/4);
  mp3_x = pos.x + (int)(1*wd/4);
  mp2_y = pos.y + (int)(3*ht/4);
  mp3_y = pos.y + (int)(1*ht/4);
  overlappingWith = 0;
  for (int i=0; i < detected_faces.size(); ++i){
    int startx = detected_faces[i].x;
    int starty = detected_faces[i].y;
    int endx = startx + this->width/detected_faces[i].scale;
    int endy = starty + this->height/detected_faces[i].scale;
    if ( (mp_x > startx ) && (mp_x < endx) && (mp_y > starty) && (mp_y < endy) ){
      overlappingWith = i;
      return true;
    }
    if ( (mp2_x > startx ) && (mp2_x < endx) && (mp_y > starty) && (mp_y < endy) ){
      overlappingWith = i;
      return true;
    }
    if ( (mp3_x > startx ) && (mp3_x < endx) && (mp_y > starty) && (mp_y < endy) ){
      overlappingWith = i;
      return true;
    }
    if ( (mp_x > startx ) && (mp_x < endx) && (mp2_y > starty) && (mp2_y < endy) ){
      overlappingWith = i;
      return true;
    }
    if ( (mp_x > startx ) && (mp_x < endx) && (mp3_y > starty) && (mp3_y < endy) ){
      overlappingWith = i;
      return true;
    }
    if ( (mp2_x > startx ) && (mp2_x < endx) && (mp2_y > starty) && (mp2_y < endy) ){
      overlappingWith = i;
      return true;
    }
    if ( (mp2_x > startx ) && (mp2_x < endx) && (mp3_y > starty) && (mp3_y < endy) ){
      overlappingWith = i;
      return true;
    }
    if ( (mp3_x > startx ) && (mp3_x < endx) && (mp2_y > starty) && (mp2_y < endy) ){
      overlappingWith = i;
      return true;
    }
    if ( (mp3_x > startx ) && (mp3_x < endx) && (mp3_y > starty) && (mp3_y < endy) ){
      overlappingWith = i;
      return true;
    }
  }
  return false;
}


int EigFaces::partition(std::vector<FacePosition>& a, int low, int high) const
{
  FacePosition right = a[high];
  int i = low -1;
  int j;
  for (j=low;j<high;j++)
    {
      if (a[j].error<=right.error)
	{
	  i++;
	  FacePosition temp = a[i];  //swap
	  a[i]=a[j];
	  a[j]=temp;
	}
    }
  FacePosition temp2 = a[i+1]; //swap
  a[i+1]= a[high];
  a[high]= temp2;
  return i+1;       //pivot
}

void EigFaces::quickSortHelper(std::vector<FacePosition>& a, int low, int high) const
{
  if (low<high)
    {
      int split = partition(a,low,high);  //finds the pivot point
      quickSortHelper(a,low, split-1); //performs quick sort recursively
      quickSortHelper(a,split+1,high);
    }
  return;
}

void EigFaces::sort(std::vector<FacePosition>& a,int low, int high) const
{
  quickSortHelper(a, low, high -1);
  return;
}

void EigFaces::morphFaces(const Face& face1, const Face& face2, double distance, Face& result) const
{
	// TODO (extra credit): MORPH along *distance* fraction of the vector from face1 to face2 by
	// interpolating between the coefficients for the two faces and reconstructing the result.
	// For example, distance 0.0 will approximate the first, while distance 1.0 will approximate the second.
	// Negative distances are ok two.

}

const Face& EigFaces::getAverage() const
{
	return average_face;
}

void EigFaces::setAverage(const Face& average)
{
	average_face=average;
}



