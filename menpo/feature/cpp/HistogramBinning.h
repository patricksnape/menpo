#pragma once
#include "WindowFeature.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <string.h>

const float pi = 3.1415926536;

using namespace std;

class HistogramBinning: public WindowFeature {
public:
	HistogramBinning(unsigned int windowHeight, unsigned int windowWidth,
	                 unsigned int numberOfChannels,
	                 unsigned int numberOfOrientationBins,
	                 unsigned int cellHeightAndWidthInPixels,
	                 unsigned int blockHeightAndWidthInCells,
	                 bool enableSignedGradients);
	virtual ~HistogramBinning();
	void apply(double *windowImage, double *descriptorVector);
	unsigned int descriptorLengthPerBlock, numberOfBlocksPerWindowHorizontally,
	             numberOfBlocksPerWindowVertically;
private:
    unsigned int numberOfOrientationBins, cellHeightAndWidthInPixels,
                 blockHeightAndWidthInCells, windowHeight, windowWidth,
                 numberOfChannels;
    bool enableSignedGradients;
};

void CreateHistogram(double *inputImage, unsigned int numberOfOrientationBins,
                     unsigned int cellHeightAndWidthInPixels,
                     unsigned int blockHeightAndWidthInCells,
                     bool signedOrUnsignedGradientsBool,
                     unsigned int imageHeight, unsigned int imageWidth,
                     unsigned int numberOfChannels, double *descriptorVector);
