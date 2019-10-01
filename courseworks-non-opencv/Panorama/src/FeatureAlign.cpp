///////////////////////////////////////////////////////////////////////////
//
// NAME
//  FeatureAlign.h -- image registration using feature matching
//
// SEE ALSO
//  FeatureAlign.h      longer description
//
// Copyright ?Richard Szeliski, 2001.  See Copyright.h for more details
// (modified for CSE576 Spring 2005)
//
///////////////////////////////////////////////////////////////////////////

#include "ImageLib/ImageLib.h"
#include "FeatureAlign.h"
#include <math.h>
#include <iostream>
/******************* TO DO *********************
* alignPair:
*	INPUT:
*		f1, f2: source feature sets
*		matches: correspondences between f1 and f2
*               *NOTE* Each match in 'matches' contains two feature ids of matching features, id1 (in f1) and id2 (in f2).
*               These ids are 1-based indices into the feature arrays,
*               so you access the appropriate features as f1[id1-1] and f2[id2-1].
*		m: motion model
*		f: focal length
*		nRANSAC: number of RANSAC iterations
*		RANSACthresh: RANSAC distance threshold
*		M: transformation matrix (output)
*	OUTPUT:
*		repeat for nRANSAC iterations:
*			choose a minimal set of feature matches
*			estimate the transformation implied by these matches
*			count the number of inliers
*		for the transformation with the maximum number of inliers,
*		compute the least squares motion estimate using the inliers,
*		and store it in M
*/
int alignPair(const FeatureSet &f1, const FeatureSet &f2,const vector<FeatureMatch> &matches, MotionModel m, float f,int nRANSAC, double RANSACthresh, CTransform3x3& M)
{
  // BEGIN TODO
  // write this entire method
  double maxX, maxY;
  int maxPos = 0;
  int maxCount = 0;
  for (int i = 0 ; i < nRANSAC ; ++i){
    double xtrans, ytrans;
    int index = rand()%(matches.size());
    Feature v1 =  f1 [ matches[index].id1 - 1 ];
    Feature v2 =  f2 [ matches[index].id2 - 1 ];
    xtrans =  v2.x - v1.x;
    ytrans = v2.y - v1.y;
    M[0][2] = xtrans;
    M[1][2] = ytrans;
    vector<int> inliers;
    int count = countInliers(f1,f2,matches,m,f,M,RANSACthresh,inliers);
    if (count > maxCount){
      maxCount = count;
      maxX = xtrans;
      maxY = ytrans;
    }
  }
  M[0][2] = maxX;
  M[1][2] = maxY;
  return maxCount;
}

/******************* TO DO *********************
* countInliers:
*	INPUT:
*		f1, f2: source feature sets
*		matches: correspondences between f1 and f2
*               *NOTE* Each match contains two feature ids of matching features, id1 (in f1) and id2 (in f2).
*               These ids are 1-based indices into the feature arrays,
*               so you access the appropriate features as f1[id1-1] and f2[id2-1].
*		m: motion model
*		f: focal length
*		M: transformation matrix
*		RANSACthresh: RANSAC distance threshold
*		inliers: inlier feature IDs
*	OUTPUT:
*		transform the matched features in f1 by M
*
*		count the number of matching features for which the transformed
*		feature f1[id1-1] is within SSD distance RANSACthresh of its match
*		f2[id2-1]
*
*		store the indices of these matches in inliers
*
*		
*/
int countInliers(const FeatureSet &f1, const FeatureSet &f2,
                 const vector<FeatureMatch> &matches, MotionModel m, float f,
                 CTransform3x3 M, double RANSACthresh, vector<int> &inliers)
{
    inliers.clear();
    int count = 0;
    for (unsigned int i=0; i<(int) matches.size(); i++) {
        // BEGIN TODO
        // determine if the ith matched feature f1[id1-1], when transformed by M,
        // is within RANSACthresh of its match in f2
        //
        // if so, increment count and append i to inliers
        //
        // *NOTE* Each match contains two feature ids of matching features, id1 and id2.
        //        These ids are 1-based indices into the feature arrays,
        //        so you access the appropriate features as f1[id1-1] and f2[id2-1].
      float x1,y1,x2,y2;
      CVector3 p;
      Feature v1 =  f1 [ matches[i].id1 - 1 ];
      Feature v2 =  f2 [ matches[i].id2 - 1 ];
      p[0] = v1.x;
      p[1] = v1.y;
      p[2] = 1;

      p = M*p;
      x1 = p[0];
      y1 = p[1];
      x2 = v2.x;
      y2 = v2.y;
      
      double dist = sqrt(pow(x2-x1,2)+pow(y2-y1,2));
      if (dist < RANSACthresh){
	inliers.push_back(i);
	count++;
      }
      // END TODO
    }
    return count;
}

/******************* TO DO *********************
* leastSquaresFit:
*	INPUT:
*		f1, f2: source feature sets
*		matches: correspondences between f1 and f2
*		m: motion model
*		f: focal length
*		inliers: inlier match indices (indexes into 'matches' array)
*		M: transformation matrix (output)
*	OUTPUT:
*		compute the transformation from f1 to f2 using only the inliers
*		and return it in M
*/
int leastSquaresFit(const FeatureSet &f1, const FeatureSet &f2,
                    const vector<FeatureMatch> &matches, MotionModel m, float f,
                    const vector<int> &inliers, CTransform3x3& M)
{
    // for project 2, the transformation is a translation and
    // only has two degrees of freedom
    //
    // therefore, we simply compute the average translation vector
    // between the feature in f1 and its match in f2 for all inliers
    double u = 0;
    double v = 0;

    for (int i=0; i<(int)inliers.size(); i++) {
        double xTrans, yTrans;

        // BEGIN TODO
        // compute the translation implied by the ith inlier match
        // and store it in (xTrans,yTrans)
	double x1,x2,y1,y2;
	x1 = f1 [ matches[inliers[i]].id1 - 1 ].x;
	y1 = f1 [ matches[inliers[i]].id1 - 1 ].y;
	
	x2 = f2 [ matches[inliers[i]].id2 - 1 ].x;
	y2 = f2 [ matches[inliers[i]].id2 - 1 ].y;

	xTrans = x2 - x1;
	yTrans = y2 - y1;
	
	// END TODO

        u += xTrans;
        v += yTrans;
    }

    u /= inliers.size();
    v /= inliers.size();

    M[0][0] = 1;
    M[0][1] = 0;
    M[0][2] = u;
    M[1][0] = 0;
    M[1][1] = 1;
    M[1][2] = v;
    M[2][0] = 0;
    M[2][1] = 0;
    M[2][2] = 1;

    return 0;
}
