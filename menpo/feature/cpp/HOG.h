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
	HOG(Py_ssize_t windowHeight,
	    Py_ssize_t windowWidth,
	    Py_ssize_t numberOfChannels,
	    unsigned int method,
	    Py_ssize_t numberOfOrientationBins,
	    Py_ssize_t cellHeightAndWidthInPixels,
	    Py_ssize_t blockHeightAndWidthInCells,
	    bool enableSignedGradients,
	    double l2normClipping);
	virtual ~HOG();
	void apply(double *windowImage, double *descriptorVector);
	Py_ssize_t descriptorLengthPerBlock,
	           numberOfBlocksPerWindowHorizontally,
	           numberOfBlocksPerWindowVertically;
private:
    Py_ssize_t numberOfOrientationBins, cellHeightAndWidthInPixels,
               blockHeightAndWidthInCells, windowHeight, windowWidth,
               numberOfChannels;
    unsigned int method;
    bool enableSignedGradients;
    double l2normClipping;
};

void ZhuRamananHOGdescriptor(double *inputImage,
                             Py_ssize_t cellHeightAndWidthInPixels,
                             Py_ssize_t imageHeight,
                             Py_ssize_t imageWidth,
                             Py_ssize_t numberOfChannels,
                             double *descriptorMatrix);
void DalalTriggsHOGdescriptor(double *inputImage,
                              Py_ssize_t numberOfOrientationBins,
                              Py_ssize_t cellHeightAndWidthInPixels,
                              Py_ssize_t blockHeightAndWidthInCells,
                              bool signedOrUnsignedGradientsBool,
                              double l2normClipping,
                              Py_ssize_t imageHeight,
                              Py_ssize_t imageWidth,
                              Py_ssize_t numberOfChannels,
                              double *descriptorVector);
