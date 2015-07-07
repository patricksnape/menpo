#pragma once
#include "WindowFeature.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define small_val 1e-6 // used to check if interpolation is needed

const double PI = 3.141592653589793238463;

using namespace std;

class LBP: public WindowFeature {
public:
	LBP(const size_t windowHeight,
	    const size_t windowWidth,
	    const size_t numberOfChannels,
	    unsigned int *radius,
	    unsigned int *samples,
	    const unsigned int numberOfRadiusSamplesCombinations,
	    const unsigned int mapping_type,
	    unsigned int *uniqueSamples,
	    unsigned int *whichMappingTable,
	    const unsigned int numberOfUniqueSamples);
	virtual ~LBP();
	void apply(const double *windowImage, double *descriptorVector);
private:
    unsigned int *samples, *whichMappingTable;
    unsigned int **mapping_tables;
    unsigned int numberOfRadiusSamplesCombinations;
    size_t windowHeight, windowWidth, numberOfChannels;
    double **samples_x_tables, **samples_y_tables;
};

void LBPdescriptor(const double *inputImage,
                   unsigned int *samples,
                   const unsigned int numberOfRadiusSamplesCombinations,
                   double **samples_x_tables,
                   double **samples_y_tables,
                   unsigned int *whichMappingTable,
                   unsigned int **mapping_tables,
                   const size_t imageHeight,
                   const size_t imageWidth,
                   const size_t numberOfChannels,
                   double *descriptorVector);
int power2(const int index);
void generate_codes_mapping_table(unsigned int *mapping_table,
                                  const unsigned int mapping_type,
                                  const unsigned int n_samples);
int count_bit_transitions(const int a, const unsigned int n_samples);
int count_bits(int n);
int leftRotate(const int num, const unsigned int len_bits,
               const unsigned int move_bits);

