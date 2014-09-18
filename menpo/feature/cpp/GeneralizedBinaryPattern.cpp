#include "GeneralizedBinaryPattern.h"

GeneralizedBinaryPattern::GeneralizedBinaryPattern(unsigned int windowHeight,
                                 unsigned int windowWidth,
                                 unsigned int numberOfChannels,
                                 unsigned int *radius, unsigned int *samples,
                                 unsigned int numberOfRadiusSamplesCombinations,
                                 unsigned int *uniqueSamples,
                                 unsigned int *whichMappingTable,
                                 unsigned int numberOfUniqueSamples) {
	unsigned int descriptorLengthPerWindow =
	                numberOfRadiusSamplesCombinations * numberOfChannels;
    this->samples = samples;
    this->whichMappingTable = whichMappingTable;
    this->numberOfRadiusSamplesCombinations = numberOfRadiusSamplesCombinations;
    this->descriptorLengthPerWindow = descriptorLengthPerWindow;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;

    // find mapping table for each unique samples value
    unsigned int **mapping_tables, i;
    mapping_tables = new unsigned int*[numberOfUniqueSamples];
    for (i = 0; i < numberOfUniqueSamples; i++) {
	    mapping_tables[i] = new unsigned int[power_2(uniqueSamples[i])];
  	    for (int j = 0; j < power_2(uniqueSamples[i]); j++)
       	    mapping_tables[i][j] = j;
	}
    this->mapping_tables = mapping_tables;

    // find coordinates of the window centre in the window reference frame
    // (axes origin in bottom left corner)
    double centre_y = (windowHeight - 1) / 2;
    double centre_x = (windowWidth - 1) / 2;

    // find samples coordinates for each radius/samples combination
    // in the window reference frame (axes origin in bottom left corner)
    double **samples_x_tables, **samples_y_tables, angle_step;
    unsigned int s;
    samples_x_tables = new double*[numberOfRadiusSamplesCombinations];
    samples_y_tables = new double*[numberOfRadiusSamplesCombinations];
    for (i = 0; i < numberOfRadiusSamplesCombinations; i++) {
        samples_x_tables[i] = new double[samples[i]];
        samples_y_tables[i] = new double[samples[i]];
        angle_step = 2 * Pi / samples[i];
        for (s = 0; s < samples[i]; s++) {
            samples_x_tables[i][s] = centre_x + radius[i] * cos(s * angle_step);
            samples_y_tables[i][s] = centre_y - radius[i] * sin(s * angle_step);
        }
    }
    this->samples_x_tables = samples_x_tables;
    this->samples_y_tables = samples_y_tables;
}

GeneralizedBinaryPattern::~GeneralizedBinaryPattern() {
    // empty memory
    delete [] mapping_tables;
    delete [] samples_x_tables;
    delete [] samples_y_tables;
}


void GeneralizedBinaryPattern::apply(double *windowImage, double *descriptorVector) {
    CreateBinaryPattern(windowImage, this->samples,
                        this->numberOfRadiusSamplesCombinations,
                        this->samples_x_tables, this->samples_y_tables,
                        this->whichMappingTable, this->mapping_tables,
                        this->windowHeight, this->windowWidth,
                        this->numberOfChannels, descriptorVector);
}


void CreateBinaryPattern(double *inputImage, unsigned int *samples,
                         unsigned int numberOfRadiusSamplesCombinations,
                         double **samples_x_tables, double **samples_y_tables,
                         unsigned int *whichMappingTable,
                         unsigned int **mapping_tables,
                         unsigned int imageHeight, unsigned int imageWidth,
                         unsigned int numberOfChannels,
                         double *descriptorVector) {
    unsigned int i, s, ch;
    int centre_y, centre_x, rx, ry, fx, fy, cx, cy, lbp_code;
    double centre_val, sample_val, tx, ty, w1, w2, w3, w4;

    // find coordinates of the window centre in the window reference frame (axes origin in bottom left corner)
    centre_y = (int)((imageHeight - 1) / 2);
    centre_x = (int)((imageWidth - 1) / 2);

    // for each radius/samples combination
    for (i = 0; i < numberOfRadiusSamplesCombinations; i++) {
        // for each channel, compute the lbp code
        for (ch = 0; ch < numberOfChannels; ch++) {
            // value of centre
            centre_val = inputImage[centre_y + centre_x * imageHeight +
                                    ch * imageHeight * imageWidth];
            lbp_code = 0;
            for (s = 0; s < samples[i]; s++) {
                // check if interpolation is needed
                rx = (int)round(samples_x_tables[i][s]);
                ry = (int)round(samples_y_tables[i][s]);
                if ( (fabs(samples_x_tables[i][s] - rx) < small_val) &&
                     (fabs(samples_y_tables[i][s] - ry) < small_val) )
                    sample_val = inputImage[ry + rx * imageHeight +
                                            ch * imageHeight * imageWidth];
                else {
                    fx = (int)floor(samples_x_tables[i][s]);
                    fy = (int)floor(samples_y_tables[i][s]);
                    cx = (int)ceil(samples_x_tables[i][s]);
                    cy = (int)ceil(samples_y_tables[i][s]);
                    tx = samples_x_tables[i][s] - fx;
                    ty = samples_y_tables[i][s] - fy;
                    // compute interpolation weights and value
                    w1 = (1 - tx) * (1 - ty);
                    w2 =      tx  * (1 - ty);
                    w3 = (1 - tx) *      ty ;
                    w4 =      tx  *      ty ;
                    sample_val = w1 * inputImage[fy + fx*imageHeight +
                                                 ch*imageHeight*imageWidth] +
                                 w2 * inputImage[fy + cx*imageHeight +
                                                 ch*imageHeight*imageWidth] +
                                 w3 * inputImage[cy + fx*imageHeight +
                                                 ch*imageHeight*imageWidth] +
                                 w4 * inputImage[cy + cx*imageHeight +
                                                 ch*imageHeight*imageWidth];
                }

                // update the lbp code
                if (sample_val >= centre_val)
                    lbp_code += power_2(s);
            }

            // store lbp code with mapping
            descriptorVector[i + ch*numberOfRadiusSamplesCombinations] =
                mapping_tables[whichMappingTable[i]][lbp_code];
        }
    }
}

int power_2(int index) {
    if (index == 0)
        return 1;
    int number = 2;
    for (int i = 1; i < index; i++)
        number = number * 2;
    return number;
}
