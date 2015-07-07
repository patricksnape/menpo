#include "ImageWindowIterator.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>

ImageWindowIterator::ImageWindowIterator(
    const double* _image,
    const size_t _imageHeight,
    const size_t _imageWidth,
    const size_t _numberOfChannels,
	const size_t _windowHeight,
	const size_t _windowWidth,
	const size_t _windowStepHorizontal,
	const size_t _windowStepVertical,
	const bool _enablePadding) :
	    imageHeight(_imageHeight), imageWidth(_imageWidth),
	    numberOfChannels(_numberOfChannels), windowHeight(_windowHeight),
	    windowWidth(_windowWidth), windowStepHorizontal(_windowStepHorizontal),
	    windowStepVertical(_windowStepVertical), enablePadding(_enablePadding),
	    image(_image) {

    size_t _numberOfWindowsHorizontally = 0,
           _numberOfWindowsVertically = 0;

    // Find number of windows
    if (!enablePadding) {
        _numberOfWindowsHorizontally = 1 + (imageWidth - windowWidth) / windowStepHorizontal;
        _numberOfWindowsVertically = 1 + (imageHeight - windowHeight) / windowStepVertical;
    }
    else {
        _numberOfWindowsHorizontally = 1 + ((imageWidth - 1) / windowStepHorizontal);
        _numberOfWindowsVertically = 1 + ((imageHeight - 1) / windowStepVertical);
    }
	this->numberOfWindowsHorizontally = _numberOfWindowsHorizontally;
	this->numberOfWindowsVertically = _numberOfWindowsVertically;
}

ImageWindowIterator::~ImageWindowIterator() {
}


void ImageWindowIterator::apply(WindowFeature *windowFeature, double *outputImage, size_t *windowsCenters) {
	long long rowCenter, rowFrom, rowTo, columnCenter, columnFrom, columnTo, i, j, k, d;

    // Initialize temporary matrices
	double* windowImage = new double[windowHeight * windowWidth * numberOfChannels];
	double* descriptorVector = new double[windowFeature->descriptorLengthPerWindow];

    // Main loop
    for (size_t windowIndexVertical = 0; windowIndexVertical < numberOfWindowsVertically; windowIndexVertical++) {
        for (size_t windowIndexHorizontal = 0; windowIndexHorizontal < numberOfWindowsHorizontally; windowIndexHorizontal++) {
            // Find window limits
            if (!enablePadding) {
                rowFrom = windowIndexVertical * windowStepVertical;
                rowTo = rowFrom + windowHeight - 1;
                rowCenter = rowFrom + (size_t)round((double)windowHeight / 2.0) - 1;
                columnFrom = windowIndexHorizontal * windowStepHorizontal;
                columnTo = columnFrom + windowWidth - 1;
                columnCenter = columnFrom + (size_t)round((double)windowWidth / 2.0) - 1;
            }
            else {
                rowCenter = windowIndexVertical * windowStepVertical;
                rowFrom = rowCenter - (size_t)round((double)windowHeight / 2.0) + 1;
                rowTo = rowFrom + windowHeight - 1;
                columnCenter = windowIndexHorizontal * windowStepHorizontal;
                columnFrom = columnCenter - (size_t)ceil((double)windowWidth / 2.0) + 1;
                columnTo = columnFrom + windowWidth - 1;
            }

            // Copy window image
			for (i = rowFrom; i <= rowTo; i++) {
				for (j = columnFrom; j <= columnTo; j++) {
					if (i < 0 || i > imageHeight-1 || j < 0 || j > imageWidth-1)
						for (k = 0; k < numberOfChannels; k++)
							windowImage[(i-rowFrom)+windowHeight*((j-columnFrom)+windowWidth*k)] = 0;
					else
						for (k=0; k < numberOfChannels; k++)
							windowImage[(i-rowFrom)+windowHeight*((j-columnFrom)+windowWidth*k)] = image[i+imageHeight*(j+imageWidth*k)];
				}
			}

            // Compute descriptor of window
            windowFeature->apply(windowImage, descriptorVector);

            // Store results
            for (d = 0; d < windowFeature->descriptorLengthPerWindow; d++)
            	outputImage[windowIndexVertical+numberOfWindowsVertically*(windowIndexHorizontal+numberOfWindowsHorizontally*d)] = descriptorVector[d];
            windowsCenters[windowIndexVertical+numberOfWindowsVertically*windowIndexHorizontal] = rowCenter;
            windowsCenters[windowIndexVertical+numberOfWindowsVertically*(windowIndexHorizontal+numberOfWindowsHorizontally)] = columnCenter;
        }
    }

    // Free temporary matrices
    delete[] windowImage;
    delete[] descriptorVector;
}

