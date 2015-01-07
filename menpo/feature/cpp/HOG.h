#pragma once
#include "WindowFeature.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES  // Required for Windows
#include <math.h>
#include <cmath>
#include <vector>
#include <string.h>

// TODO: Change to using math.h M_PI
const float pi = 3.1415926536;
#define EPS 0.0001  // Small value, used to avoid division by zero
#define ZHU_RAMANAN 2  // ENUM type defines to choose the algorithm type
#define DALAL_TRIGGS 1

using namespace std;

#define inline_min(x, y) ((x) <= (y) ? (x) : (y))
#define inline_max(x, y) ((x) <= (y) ? (y) : (x))

class HOG: public WindowFeature {
public:
	HOG(size_t windowHeight,
	    size_t windowWidth,
	    size_t numberOfChannels,
	    unsigned int method,
	    size_t numberOfOrientationBins,
	    size_t cellHeightAndWidthInPixels,
	    size_t blockHeightAndWidthInCells,
	    bool enableSignedGradients,
	    double l2normClipping);
	virtual ~HOG();
	void apply(double *windowImage, double *descriptorVector);
	size_t descriptorLengthPerBlock,
	           numberOfBlocksPerWindowHorizontally,
	           numberOfBlocksPerWindowVertically;
private:
    size_t numberOfOrientationBins, cellHeightAndWidthInPixels,
               blockHeightAndWidthInCells, windowHeight, windowWidth,
               numberOfChannels;
    unsigned int method;
    bool enableSignedGradients;
    double l2normClipping;
};

void ZhuRamananHOGdescriptor(double *inputImage,
                             size_t cellHeightAndWidthInPixels,
                             size_t imageHeight,
                             size_t imageWidth,
                             size_t numberOfChannels,
                             double *descriptorMatrix);
void DalalTriggsHOGdescriptor(double *inputImage,
                              size_t numberOfOrientationBins,
                              size_t cellHeightAndWidthInPixels,
                              size_t blockHeightAndWidthInCells,
                              bool signedOrUnsignedGradientsBool,
                              double l2normClipping,
                              size_t imageHeight,
                              size_t imageWidth,
                              size_t numberOfChannels,
                              double *descriptorVector);
