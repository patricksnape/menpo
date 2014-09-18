#pragma once
#include "WindowFeature.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <string.h>

const float P = 3.1415926536;

using namespace std;

class GeneralizedHistogramBinning: public WindowFeature {
public:
	GeneralizedHistogramBinning(unsigned int windowHeight,
	                            unsigned int windowWidth,
	                            unsigned int numberOfChannels,
	                            unsigned int numberOfOrientationBins,
	                            unsigned int cellHeightAndWidthInPixels,
	                            unsigned int blockHeightAndWidthInCells,
	                            bool enableSignedGradients,
	                            double l2normClipping);
	virtual ~GeneralizedHistogramBinning();
	void apply(double *windowImage, double *descriptorVector);
	unsigned int descriptorLengthPerBlock, numberOfBlocksPerWindowHorizontally,
	             numberOfBlocksPerWindowVertically;
private:
    unsigned int numberOfOrientationBins, cellHeightAndWidthInPixels,
                 blockHeightAndWidthInCells, windowHeight, windowWidth,
                 numberOfChannels;
    bool enableSignedGradients;
    double l2normClipping;
};

void CreateHistogram(double *inputImage, unsigned int numberOfOrientationBins,
                     unsigned int cellHeightAndWidthInPixels,
                     unsigned int blockHeightAndWidthInCells,
                     bool signedOrUnsignedGradientsBool, double l2normClipping,
                     unsigned int imageHeight, unsigned int imageWidth,
                     unsigned int numberOfChannels, double *descriptorVector);
