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
	HOG(const size_t windowHeight,
	    const size_t windowWidth,
	    const size_t numberOfChannels,
	    const unsigned int method,
	    const size_t numberOfOrientationBins,
	    const size_t cellHeightAndWidthInPixels,
	    const size_t blockHeightAndWidthInCells,
	    const bool enableSignedGradients,
	    const double l2normClipping);
	virtual ~HOG();
	void apply(const double *windowImage, double *descriptorVector);
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

void ZhuRamananHOGdescriptor(const double *inputImage,
                             const size_t cellHeightAndWidthInPixels,
                             const size_t imageHeight,
                             const size_t imageWidth,
                             const size_t numberOfChannels,
                             double *descriptorMatrix);
void DalalTriggsHOGdescriptor(const double *inputImage,
                              const size_t numberOfOrientationBins,
                              const size_t cellHeightAndWidthInPixels,
                              const size_t blockHeightAndWidthInCells,
                              const bool signedOrUnsignedGradientsBool,
                              const double l2normClipping,
                              const size_t imageHeight,
                              const size_t imageWidth,
                              const size_t numberOfChannels,
                              double *descriptorVector);
