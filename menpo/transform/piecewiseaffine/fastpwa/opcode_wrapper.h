#include <cmath>
#include "opcode/Opcode.h"

using namespace Opcode;

struct Mesh {
    IceMaths::Point *cVertices;
    IceMaths::IndexedTriangle *cIndices;
};


void convertfv(const float* v, const int n_v,
               const int* f, const int n_f,
               Mesh* fv);

Model* createOpcode(const float* vertices, const int n_v,
                    const int* faces, const int n_f);

void intersect(const Model* opcode, const float* rayo, const int n_c,
               int* indices, float* alphas, float* betas);
