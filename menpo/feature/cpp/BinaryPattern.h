#pragma once
#include "WindowFeature.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define small_val 1e-6 //used to check if interpolation is needed

const double PI = 3.141592653589793238463;

using namespace std;

class BinaryPattern: public WindowFeature {
public:
	BinaryPattern(unsigned int windowHeight, unsigned int windowWidth,
	              unsigned int numberOfChannels, unsigned int *radius,
	              unsigned int *samples,
	              unsigned int numberOfRadiusSamplesCombinations,
	              unsigned int *uniqueSamples,
	              unsigned int *whichMappingTable,
	              unsigned int numberOfUniqueSamples);
	virtual ~BinaryPattern();
	void apply(double *windowImage, double *descriptorVector);
private:
    unsigned int *samples, *whichMappingTable, **mapping_tables;
    unsigned int numberOfRadiusSamplesCombinations, windowHeight, windowWidth,
                 numberOfChannels;
    double **samples_x_tables, **samples_y_tables;
};

void CreateBinaryPattern(double *inputImage, unsigned int *samples,
                         unsigned int numberOfRadiusSamplesCombinations,
                         double **samples_x_tables, double **samples_y_tables,
                         unsigned int *whichMappingTable,
                         unsigned int **mapping_tables,
                         unsigned int imageHeight, unsigned int imageWidth,
                         unsigned int numberOfChannels,
                         double *descriptorVector);
int power2(int index);
