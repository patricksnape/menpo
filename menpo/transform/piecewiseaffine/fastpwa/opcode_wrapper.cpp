#include "opcode_wrapper.h"

using namespace Opcode;

void convertfv(const float* v, const int n_v,
               const int* f, const int n_f,
               Mesh* fv) {
    int ki = 0, offset = 0;

    IceMaths::IndexedTriangle* fc = new IceMaths::IndexedTriangle[n_f];
    for (ki = 0; ki < n_f; ki++) {
        offset = ki * 3;
        fc[ki] = IceMaths::IndexedTriangle(f[offset],
                                           f[offset + 1],
                                           f[offset + 2]);
    }

    IceMaths::Point *vc = new IceMaths::Point[n_v];
    for (ki = 0; ki < n_v; ki++) {
        offset = ki * 2;
        vc[ki] = IceMaths::Point(v[offset],
                                 v[offset + 1],
                                 0.0);
    }

    fv->cIndices = fc;
    fv->cVertices = vc;
}

Model* createOpcode(const float* vertices, const int n_v,
                    const int* faces, const int n_f) {
    Mesh* fv = new Mesh;
    convertfv(vertices, n_v, faces, n_f, fv);

    // Create the tree
    Model* opcode = new Model;
    MeshInterface *cMesh = new MeshInterface;
    cMesh->SetInterfaceType(MESH_TRIANGLE);
    cMesh->SetNbTriangles(n_f);
    cMesh->SetNbVertices(n_v);
    cMesh->SetPointers(fv->cIndices, fv->cVertices);

    OPCODECREATE OPCC;
    OPCC.mIMesh = cMesh;

    BuildSettings cBS;
    cBS.mRules = SPLIT_SPLATTER_POINTS | SPLIT_GEOM_CENTER;
    OPCC.mSettings = cBS;
    OPCC.mNoLeaf = true;
    OPCC.mQuantized = false;
    // Set to True when debugging.
    OPCC.mKeepOriginal = false;
    bool status = opcode->Build(OPCC);

//    if (!status) {
//      mexErrMsgTxt("Error when making tree.");
//    }

//    plhs[0] = convertPtr2Mat<Model>(opcode);
    return opcode;
}

void intersect(const Model* opcode, const float* rayo,
               const int n_c,
               int* indices, float* alphas, float* betas) {
    // Req. aabb tree handle, ray start (3 x n_c) and ray direction (3 x n_c)
    RayCollider RC;
    RC.SetFirstContact(false);
    RC.SetClosestHit(true);
    RC.SetCulling(false);
    //RC.SetMaxDist(inf);

    CollisionFaces CF;
    RC.SetDestination(&CF);

    bool status, hit;
    int i = 0, offset = 0;
    IceMaths::Point cStart;
    IceMaths::Ray cRay;
    IceMaths::Point cDir = IceMaths::Point(0.0, 0.0, -1.0);

    for (i = 0; i < n_c; i++) {
        offset = i * 2;
        IceMaths::Point cStart = IceMaths::Point(rayo[offset],
                                                 rayo[offset + 1],
                                                 1.0);
        IceMaths::Ray cRay = IceMaths::Ray(cStart, cDir);
        static udword Cache;
        status = RC.Collide(cRay, *opcode, NULL, &Cache);
        //if (!status) mexErrMsgTxt("Error when hitting.");

        hit = RC.GetContactStatus();

        const CollisionFace* colFaces = CF.GetFaces();
        indices[i] = hit ? colFaces[0].mFaceID : -1;
        alphas[i] = hit ? colFaces[0].mU : NAN;
        betas[i] = hit ? colFaces[0].mV : NAN;
    }
}
