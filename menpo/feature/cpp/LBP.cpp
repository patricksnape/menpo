#include "LBP.h"

LBP::LBP(const size_t windowHeight,
         const size_t windowWidth,
         const size_t numberOfChannels,
         unsigned int *radius,
         unsigned int *samples,
         const unsigned int numberOfRadiusSamplesCombinations,
         const unsigned int mapping_type,
         unsigned int *uniqueSamples,
         unsigned int *whichMappingTable,
         const unsigned int numberOfUniqueSamples) {

	const unsigned int descriptorLengthPerWindow = numberOfRadiusSamplesCombinations * numberOfChannels;

    this->samples = samples;
    this->whichMappingTable = whichMappingTable;
    this->numberOfRadiusSamplesCombinations = numberOfRadiusSamplesCombinations;
    this->descriptorLengthPerWindow = descriptorLengthPerWindow;
    this->windowHeight = windowHeight;
    this->windowWidth = windowWidth;
    this->numberOfChannels = numberOfChannels;

    // find mapping table for each unique samples value
    unsigned int **mapping_tables = new unsigned int*[numberOfUniqueSamples];
    for (unsigned int i = 0; i < numberOfUniqueSamples; i++) {
	    mapping_tables[i] = new unsigned int[power2(uniqueSamples[i])];
	    if (mapping_type != 0) {
    	    generate_codes_mapping_table(mapping_tables[i], mapping_type,
    	                                 uniqueSamples[i]);
    	} else {
    	    for (int j = 0; j < power2(uniqueSamples[i]); j++) {
        	    mapping_tables[i][j] = j;
        	}
        }
	}
    this->mapping_tables = mapping_tables;

    // find coordinates of the window centre in the window reference frame
    // (axes origin in bottom left corner)
    const double centre_y = (windowHeight - 1) / 2.0;
    const double centre_x = (windowWidth - 1) / 2.0;

    // find samples coordinates for each radius/samples combination
    // in the window reference frame (axes origin in bottom left corner)
    double **samples_x_tables = new double*[numberOfRadiusSamplesCombinations];
    double **samples_y_tables = new double*[numberOfRadiusSamplesCombinations];
    for (unsigned int i = 0; i < numberOfRadiusSamplesCombinations; i++) {
        samples_x_tables[i] = new double[samples[i]];
        samples_y_tables[i] = new double[samples[i]];
        const double angle_step = 2 * PI / samples[i];
        for (unsigned int s = 0; s < samples[i]; s++) {
            samples_x_tables[i][s] = centre_x + radius[i] * cos(s * angle_step);
            samples_y_tables[i][s] = centre_y - radius[i] * sin(s * angle_step);
        }
    }
    this->samples_x_tables = samples_x_tables;
    this->samples_y_tables = samples_y_tables;
}

LBP::~LBP() {
    // empty memory
    delete[] mapping_tables;
    delete[] samples_x_tables;
    delete[] samples_y_tables;
}


void LBP::apply(const double *windowImage, double *descriptorVector) {
    LBPdescriptor(windowImage, this->samples,
                  this->numberOfRadiusSamplesCombinations,
                  this->samples_x_tables, this->samples_y_tables,
                  this->whichMappingTable, this->mapping_tables,
                  this->windowHeight, this->windowWidth,
                  this->numberOfChannels, descriptorVector);
}


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
                   double *descriptorVector) {
    int rx, ry, fx, fy, cx, cy, lbp_code;
    double centre_val, sample_val, tx, ty, w1, w2, w3, w4;

    // find coordinates of the window centre in the window reference frame (axes origin in bottom left corner)
    const size_t centre_y = (int)((imageHeight - 1) / 2);
    const size_t centre_x = (int)((imageWidth - 1) / 2);

    // for each radius/samples combination
    for (unsigned int i = 0; i < numberOfRadiusSamplesCombinations; i++) {
        // for each channel, compute the lbp code
        for (size_t ch = 0; ch < numberOfChannels; ch++) {
            // value of centre
            centre_val = inputImage[centre_y + centre_x * imageHeight +
                                    ch * imageHeight * imageWidth];
            lbp_code = 0;
            for (unsigned int s = 0; s < samples[i]; s++) {
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
                    lbp_code += power2(s);
            }

            // store lbp code with mapping
            descriptorVector[i + ch*numberOfRadiusSamplesCombinations] =
                mapping_tables[whichMappingTable[i]][lbp_code];
        }
    }
}

int power2(const int index) {
    if (index == 0)
        return 1;
    int number = 2;
    for (int i = 1; i < index; i++)
        number = number * 2;
    return number;
}

void generate_codes_mapping_table(unsigned int *mapping_table,
                                  const unsigned int mapping_type,
                                  const unsigned int n_samples) {
    unsigned int newMax = 0;

    if (mapping_type == 1) {
        // uniform-2
        newMax = n_samples * (n_samples - 1) + 3;
        int index = 0, num_trans = 0;
        for (unsigned int c = 0; c < power2(n_samples); c++) {
            // number of 1->0 and 0->1 transitions in a binary string x is
            // equal to the number of 1-bits in XOR(x, rotate_left(x))
            num_trans = count_bit_transitions(c, n_samples);
            if (num_trans <= 2) {
                mapping_table[c] = index;
                index += 1;
            } else {
                mapping_table[c] = newMax - 1;
            }
        }
    } else if (mapping_type == 2) {
        // rotation invariant
        int rm = 0, r = 0;
        int *tmp_map = new int[power2(n_samples)];
        for (unsigned int c = 0; c < power2(n_samples); c++) {
            tmp_map[c] = -1;
        }
        newMax = 0;
        for (unsigned int c = 0; c < power2(n_samples); c++) {
            rm = c;
            r = c;
            for (unsigned int j = 1; j < n_samples; j++) {
                r = leftRotate(r, n_samples, 1);
                if (r < rm) {
                    rm = r;
                }
            }
            if (tmp_map[rm] < 0) {
                tmp_map[rm] = newMax;
                newMax += 1;
            }
            mapping_table[c] = tmp_map[rm];
        }
        delete[] tmp_map;
    } else if (mapping_type == 3) {
        // rotation invariant and uniform-2
        int num_trans = 0;
        newMax = n_samples + 2;
        for (unsigned int c = 0; c < power2(n_samples); c++) {
            // number of 1->0 and 0->1 transitions in a binary string x is
            // equal to the number of 1-bits in XOR(x, rotate_left(x))
            num_trans = count_bit_transitions(c, n_samples);
            if (num_trans <= 2) {
                mapping_table[c] = count_bits(c);
            } else {
                mapping_table[c] = n_samples + 1;
            }
        }
    }
}

int count_bit_transitions(const int a, const unsigned int n_samples) {
    const int b = a >> 1; // sign-extending shift properly counts bits at the ends
    const int c = a ^ b;  // xor marks bits that are not the same as their neighbors on the left
    if (a >= power2(n_samples - 1)) {
        return count_bits(c) - 1; // count number of set bits in c
    } else {
        return count_bits(c); // count number of set bits in c
    }
}

int count_bits(int n) {
    int c = 0;
    for (c = 0; n; c++) {
        n &= n - 1; // clear the least significant bit set
     }
    return c;
}

int leftRotate(const int num, const unsigned int len_bits,
               const unsigned int move_bits) {
   // In num<<move_bits, last move_bits bits are 0. To put first 3 bits of num
   // at last, do bitwise or of num<<move_bits with num >>(len_bits - move_bits)
   return ((num << move_bits % len_bits) & (power2(len_bits) - 1)) |
          ((num & (power2(len_bits) - 1)) >> (len_bits - (move_bits % len_bits)));
}
