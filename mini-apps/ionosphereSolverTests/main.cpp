#include <iostream>
#include <sys/time.h>
#include "vlsv_writer.h"
#include "vlsv_reader_parallel.h"
#include "../../sysboundary/ionosphere.h"
#include "../../object_wrapper.h"
#include "../../datareduction/datareductionoperator.h"
#include "../../iowrite.h"
#include "../../ioread.h"
#include "../../velocity_mesh_parameters.h"
#include "../../logger.h"
#include "../../open_bucket_hashtable.h"

#include <Eigen/Sparse>
#include <Eigen/Geometry>

#define NODE_CONSTRAINT_REDUCTION -1
#define ELEMENT_CONSTRAINT_REDUCTION -1

using namespace std;
using namespace SBC;
using namespace vlsv;

Logger logFile,diagnostic;
int globalflags::bailingOut=0;
bool globalflags::writeRestart=false;
bool globalflags::writeRecover=false;
bool globalflags::balanceLoad=false;
bool globalflags::doRefine=false;
bool globalflags::ionosphereJustSolved = false;
ObjectWrapper objectWrapper;
ObjectWrapper& getObjectWrapper() {
   return objectWrapper;
}

// Dummy implementations of some functions to make things compile
std::vector<CellID> localCellDummy;
const std::vector<CellID>& getLocalCells() { return localCellDummy; }
void deallocateRemoteCellBlocks(dccrg::Dccrg<spatial_cell::SpatialCell, dccrg::Cartesian_Geometry, std::tuple<>, std::tuple<> >&) {};
void updateRemoteVelocityBlockLists(dccrg::Dccrg<spatial_cell::SpatialCell, dccrg::Cartesian_Geometry, std::tuple<>, std::tuple<> >&, unsigned int, unsigned int) {};
void recalculateLocalCellsCache(const dccrg::Dccrg<spatial_cell::SpatialCell, dccrg::Cartesian_Geometry, std::tuple<>, std::tuple<> >&) {};
SysBoundary::SysBoundary() {}
SysBoundary::~SysBoundary() {}

// Element Barycentre
Eigen::Vector3d getElementBarycentre(SphericalTriGrid& grid, uint32_t el) {
   Eigen::Vector3d barycentre(0,0,0);

   SphericalTriGrid::Element& element = grid.elements[el];
   for(uint i=0; i<3; i++) {
      Eigen::Vector3d corner(grid.nodes[element.corners[i]].x.data());

      barycentre += corner;
   }
   barycentre /= 3.;

   return barycentre;
}

Eigen::Vector3d getElementNormal(SphericalTriGrid& grid, uint32_t el) {
   Eigen::Vector3d normal(0,0,0);

   SphericalTriGrid::Element& element = grid.elements[el];
   uint32_t corner1 = element.corners[0];
   uint32_t corner2 = element.corners[1];
   uint32_t corner3 = element.corners[2];

   Eigen::Vector3d a(grid.nodes[corner1].x.data());
   Eigen::Vector3d b(grid.nodes[corner2].x.data());
   Eigen::Vector3d c(grid.nodes[corner3].x.data());

   Eigen::Vector3d edge1 = b - a;
   Eigen::Vector3d edge2 = c - a;

   normal = edge1.cross(edge2);

   normal.normalize();

   if(normal.dot(getElementBarycentre(grid, el)) < 0) {
      normal *= -1.;
   }

   return normal;
}

Real getDualPolygonArea(SphericalTriGrid& grid, uint gridNode){
   Real A = 0.;
   for(uint i = 0; i < grid.nodes[gridNode].numTouchingElements; i++){
      uint32_t gridEl = grid.nodes[gridNode].touchingElements[i];
      Eigen::Vector3d nodePosition(grid.nodes[gridNode].x.data());
      SphericalTriGrid::Element& element = grid.elements[gridEl];

      int localC=0,localI=0,localJ=0;
      for(int c=0; c < 3; c++) {
         if(element.corners[c] == gridNode) {
            localC = c;
            localI = (c+1)%3;
            localJ = (c+2)%3;
            break;
         }
      }

      uint otherElementi = ionosphereGrid.findElementNeighbour(gridEl, localC, localI);
      uint otherElementj = ionosphereGrid.findElementNeighbour(gridEl, localC, localJ);

      Eigen::Vector3d centerm = getElementBarycentre(ionosphereGrid, gridEl);
      Eigen::Vector3d centeri = getElementBarycentre(ionosphereGrid, otherElementi);
      Eigen::Vector3d centerj = getElementBarycentre(ionosphereGrid, otherElementj);

      Eigen::Vector3d normalm = getElementNormal(ionosphereGrid, gridEl);
      Eigen::Vector3d normali = getElementNormal(ionosphereGrid, otherElementi);
      Eigen::Vector3d normalj = getElementNormal(ionosphereGrid, otherElementj);

      Eigen::Vector3d rotatedCenteri = nodePosition + Eigen::Quaternion<Real>::FromTwoVectors(normali, normalm).toRotationMatrix() * (centeri - nodePosition);
      Eigen::Vector3d rotatedCenterj = nodePosition + Eigen::Quaternion<Real>::FromTwoVectors(normalj, normalm).toRotationMatrix() * (centerj - nodePosition);

      Eigen::Vector3d edgeNodeM = nodePosition - centerm;
      Eigen::Vector3d edgeNodeI = nodePosition - rotatedCenteri;
      Eigen::Vector3d edgeNodeJ = nodePosition - rotatedCenterj;

      Real areaMI = edgeNodeM.cross(edgeNodeI).norm() / 2.;
      Real areaMJ = edgeNodeM.cross(edgeNodeJ).norm() / 2.;

      // Double counting
      A += (areaMI + areaMJ) / 2.;

   }
   return A;
}

// Calculate neighbor's Barycentre and dual polygon - edge - intersection point.
std::tuple<Eigen::Vector3d, Eigen::Vector3d> getConnectingSegmentLengths(SphericalTriGrid& grid, uint32_t el1, uint32_t el2) {
   SphericalTriGrid::Element& element1 = grid.elements[el1];
   SphericalTriGrid::Element& element2 = grid.elements[el2];

   Eigen::Vector3d barycentre1 = getElementBarycentre(grid, el1);
   Eigen::Vector3d barycentre2 = getElementBarycentre(grid, el2);

   // Get common edge to these two elements
   for(uint i=0; i<3; i++) {
      if(element1.corners[i] == element2.corners[0] ||
         element1.corners[i] == element2.corners[1] ||
         element1.corners[i] == element2.corners[2]) {
         for(uint j=0; j<3; j++) {
            if(i != j && (element1.corners[j] == element2.corners[0] ||
                          element1.corners[j] == element2.corners[1] ||
                          element1.corners[j] == element2.corners[2])) {

                  uint corner1 = element1.corners[i];
                  uint corner2 = element1.corners[j];

                  Eigen::Vector3d normal1 = getElementNormal(grid, el1);
                  Eigen::Vector3d normal2 = getElementNormal(grid, el2);

                  Eigen::Vector3d rotatedBarycentre2 =  Eigen::Vector3d(grid.nodes[corner1].x.data()) +
                                                         Eigen::Quaternion<Real>::FromTwoVectors(normal2, normal1).toRotationMatrix() *
                                                         (barycentre2 - Eigen::Vector3d(grid.nodes[corner1].x.data()));

                  Eigen::Vector3d corner1Position(grid.nodes[corner1].x.data());
                  Eigen::Vector3d corner2Position(grid.nodes[corner2].x.data());

                  Eigen::Vector3d barycentre1ToBarycentre2 = (rotatedBarycentre2 - barycentre1).normalized();
                  Eigen::Vector3d corner1ToCorner2 = (corner2Position - corner1Position).normalized();

                  // Get intersection of line between barycenters and line between corners
                  Eigen::Matrix<double, 3, 2> A;
                  A.col(0) = barycentre1ToBarycentre2;
                  A.col(1) = - corner1ToCorner2;
                  Eigen::Vector3d b = corner1Position - barycentre1;
                  Eigen::Vector2d t = A.colPivHouseholderQr().solve(b);
                  Eigen::Vector3d intersection = barycentre1 + t(0) * barycentre1ToBarycentre2;

                  return std::make_tuple(barycentre2, intersection);
            }
         }
      }
   }

   // Not found, something went bananas.
   abort();
}

// Ionosoheric Sigma calculation function coefficients from
// Juusola et al. 2025
// Note: MLT is in hours
const Real c1p = 0.351;
const Real c2p = 0.697;
const Real c3p = 0.707;
const Real c1h = 0.720;
const Real c2h = 0.617;
const Real c3h = 0.846;
std::function<Real(Real)> c4P = [](Real MLT) {
   const Real values[] = {
      0.272, // 00
      0.212, // 01
      0.268, // 02
      0.357, // 03
      0.241, // 04
      0.225, // 05
      0.283, // 06
      0.621, // 07
      1.185, // 08
      1.643, // 09
      1.756, // 10
      1.562, // 11
      1.421, // 12
      1.327, // 13
      1.001, // 14
      0.959, // 15
      0.672, // 16
      0.300, // 17
      0.150, // 18
      0.241, // 19
      0.253, // 20
      0.340, // 21
      0.391, // 22
      0.413  // 23
   };
   int sector = MLT;
   Real interpolant = MLT - sector;
   return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%24];
};

std::function<Real(Real)> c5P = [](Real MLT) {
   const Real values[] = {
      0.564, // 00
      0.602, // 01
      0.565, // 02
      0.507, // 03
      0.578, // 04
      0.598, // 05
      0.538, // 06
      0.338, // 07
      0.175, // 08
      0.075, // 09
      0.011, // 10
      0.000, // 11
      0.000, // 12
      0.037, // 13
      0.164, // 14
      0.222, // 15
      0.314, // 16
      0.470, // 17
      0.645, // 18
      0.581, // 19
      0.587, // 20
      0.533, // 21
      0.504, // 22
      0.498  // 23
   };
   int sector = MLT;
   Real interpolant = MLT - sector;
   return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%24];
};

std::function<Real(Real)> c4H = [](Real MLT) {
   const Real values[] = {
      0.184, // 00
      0.248, // 01
      0.256, // 02
      0.343, // 03
      0.344, // 04
      0.341, // 05
      0.285, // 06
      0.676, // 07
      1.617, // 08
      2.610, // 09
      2.818, // 10
      2.962, // 11
      2.579, // 12
      2.049, // 13
      1.367, // 14
      0.701, // 15
      0.277, // 16
      0.141, // 17
      0.081, // 18
      0.172, // 19
      0.178, // 20
      0.223, // 21
      0.322, // 22
      0.373  // 23
   };
   int sector = MLT;
   Real interpolant = MLT - sector;
   return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%24];
};

std::function<Real(Real)> c5H = [](Real MLT) {
   const Real values[] = {
      0.757, // 00
      0.702, // 01
      0.705, // 02
      0.649, // 03
      0.652, // 04
      0.662, // 05
      0.683, // 06
      0.453, // 07
      0.236, // 08
      0.115, // 09
      0.079, // 10
      0.042, // 11
      0.035, // 12
      0.030, // 13
      0.126, // 14
      0.306, // 15
      0.529, // 16
      0.653, // 17
      0.811, // 18
      0.710, // 19
      0.753, // 20
      0.721, // 21
      0.657, // 22
      0.635  // 23
   };
   int sector = MLT;
   Real interpolant = MLT - sector;
   return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%24];
};

// Tabulated chapman function for atmospheric absorption of EUV values, from Laundal et al. 2024
const static Real chapman_euv_table[1201] = {
   1.00000e+00, 9.99998e-01, 9.99994e-01, 9.99986e-01, 9.99976e-01, 9.99962e-01, 9.99946e-01, 9.99926e-01, 9.99904e-01, 9.99878e-01,
   9.99850e-01, 9.99818e-01, 9.99784e-01, 9.99746e-01, 9.99706e-01, 9.99662e-01, 9.99616e-01, 9.99566e-01, 9.99514e-01, 9.99458e-01,
   9.99400e-01, 9.99338e-01, 9.99274e-01, 9.99206e-01, 9.99136e-01, 9.99062e-01, 9.98986e-01, 9.98906e-01, 9.98824e-01, 9.98738e-01,
   9.98650e-01, 9.98558e-01, 9.98464e-01, 9.98366e-01, 9.98266e-01, 9.98162e-01, 9.98056e-01, 9.97946e-01, 9.97834e-01, 9.97719e-01,
   9.97600e-01, 9.97479e-01, 9.97354e-01, 9.97227e-01, 9.97096e-01, 9.96963e-01, 9.96827e-01, 9.96687e-01, 9.96545e-01, 9.96399e-01,
   9.96251e-01, 9.96100e-01, 9.95945e-01, 9.95788e-01, 9.95628e-01, 9.95464e-01, 9.95298e-01, 9.95129e-01, 9.94957e-01, 9.94781e-01,
   9.94603e-01, 9.94422e-01, 9.94238e-01, 9.94051e-01, 9.93860e-01, 9.93667e-01, 9.93471e-01, 9.93272e-01, 9.93070e-01, 9.92865e-01,
   9.92657e-01, 9.92446e-01, 9.92232e-01, 9.92015e-01, 9.91795e-01, 9.91572e-01, 9.91346e-01, 9.91117e-01, 9.90885e-01, 9.90651e-01,
   9.90413e-01, 9.90172e-01, 9.89928e-01, 9.89682e-01, 9.89432e-01, 9.89179e-01, 9.88924e-01, 9.88665e-01, 9.88404e-01, 9.88139e-01,
   9.87872e-01, 9.87601e-01, 9.87328e-01, 9.87051e-01, 9.86772e-01, 9.86490e-01, 9.86205e-01, 9.85916e-01, 9.85625e-01, 9.85331e-01,
   9.85034e-01, 9.84734e-01, 9.84431e-01, 9.84125e-01, 9.83817e-01, 9.83505e-01, 9.83190e-01, 9.82872e-01, 9.82552e-01, 9.82228e-01,
   9.81901e-01, 9.81572e-01, 9.81240e-01, 9.80904e-01, 9.80566e-01, 9.80225e-01, 9.79880e-01, 9.79533e-01, 9.79183e-01, 9.78830e-01,
   9.78474e-01, 9.78116e-01, 9.77754e-01, 9.77389e-01, 9.77021e-01, 9.76651e-01, 9.76277e-01, 9.75901e-01, 9.75522e-01, 9.75139e-01,
   9.74754e-01, 9.74366e-01, 9.73975e-01, 9.73581e-01, 9.73184e-01, 9.72784e-01, 9.72382e-01, 9.71976e-01, 9.71567e-01, 9.71156e-01,
   9.70742e-01, 9.70324e-01, 9.69904e-01, 9.69481e-01, 9.69055e-01, 9.68626e-01, 9.68195e-01, 9.67760e-01, 9.67322e-01, 9.66882e-01,
   9.66438e-01, 9.65992e-01, 9.65543e-01, 9.65091e-01, 9.64636e-01, 9.64178e-01, 9.63718e-01, 9.63254e-01, 9.62787e-01, 9.62318e-01,
   9.61846e-01, 9.61371e-01, 9.60893e-01, 9.60412e-01, 9.59928e-01, 9.59441e-01, 9.58952e-01, 9.58460e-01, 9.57964e-01, 9.57466e-01,
   9.56965e-01, 9.56461e-01, 9.55955e-01, 9.55445e-01, 9.54933e-01, 9.54417e-01, 9.53899e-01, 9.53378e-01, 9.52855e-01, 9.52328e-01,
   9.51798e-01, 9.51266e-01, 9.50731e-01, 9.50193e-01, 9.49652e-01, 9.49108e-01, 9.48561e-01, 9.48012e-01, 9.47460e-01, 9.46904e-01,
   9.46347e-01, 9.45786e-01, 9.45222e-01, 9.44656e-01, 9.44087e-01, 9.43514e-01, 9.42940e-01, 9.42362e-01, 9.41781e-01, 9.41198e-01,
   9.40612e-01, 9.40023e-01, 9.39431e-01, 9.38837e-01, 9.38239e-01, 9.37639e-01, 9.37036e-01, 9.36430e-01, 9.35822e-01, 9.35210e-01,
   9.34596e-01, 9.33979e-01, 9.33359e-01, 9.32737e-01, 9.32111e-01, 9.31483e-01, 9.30852e-01, 9.30219e-01, 9.29582e-01, 9.28943e-01,
   9.28301e-01, 9.27656e-01, 9.27009e-01, 9.26358e-01, 9.25705e-01, 9.25049e-01, 9.24391e-01, 9.23729e-01, 9.23065e-01, 9.22398e-01,
   9.21729e-01, 9.21056e-01, 9.20381e-01, 9.19703e-01, 9.19022e-01, 9.18339e-01, 9.17653e-01, 9.16964e-01, 9.16273e-01, 9.15578e-01,
   9.14881e-01, 9.14181e-01, 9.13479e-01, 9.12774e-01, 9.12066e-01, 9.11355e-01, 9.10642e-01, 9.09925e-01, 9.09207e-01, 9.08485e-01,
   9.07761e-01, 9.07034e-01, 9.06304e-01, 9.05572e-01, 9.04837e-01, 9.04099e-01, 9.03359e-01, 9.02616e-01, 9.01870e-01, 9.01121e-01,
   9.00370e-01, 8.99616e-01, 8.98860e-01, 8.98100e-01, 8.97338e-01, 8.96574e-01, 8.95807e-01, 8.95037e-01, 8.94264e-01, 8.93489e-01,
   8.92711e-01, 8.91930e-01, 8.91147e-01, 8.90361e-01, 8.89573e-01, 8.88782e-01, 8.87988e-01, 8.87191e-01, 8.86392e-01, 8.85591e-01,
   8.84786e-01, 8.83979e-01, 8.83170e-01, 8.82357e-01, 8.81542e-01, 8.80725e-01, 8.79905e-01, 8.79082e-01, 8.78257e-01, 8.77429e-01,
   8.76598e-01, 8.75765e-01, 8.74929e-01, 8.74091e-01, 8.73250e-01, 8.72406e-01, 8.71560e-01, 8.70712e-01, 8.69860e-01, 8.69006e-01,
   8.68150e-01, 8.67291e-01, 8.66429e-01, 8.65565e-01, 8.64698e-01, 8.63829e-01, 8.62957e-01, 8.62082e-01, 8.61205e-01, 8.60326e-01,
   8.59444e-01, 8.58559e-01, 8.57672e-01, 8.56782e-01, 8.55890e-01, 8.54995e-01, 8.54097e-01, 8.53198e-01, 8.52295e-01, 8.51390e-01,
   8.50483e-01, 8.49573e-01, 8.48660e-01, 8.47745e-01, 8.46828e-01, 8.45907e-01, 8.44985e-01, 8.44060e-01, 8.43132e-01, 8.42202e-01,
   8.41270e-01, 8.40335e-01, 8.39397e-01, 8.38457e-01, 8.37515e-01, 8.36570e-01, 8.35622e-01, 8.34672e-01, 8.33720e-01, 8.32765e-01,
   8.31808e-01, 8.30848e-01, 8.29886e-01, 8.28921e-01, 8.27954e-01, 8.26984e-01, 8.26012e-01, 8.25038e-01, 8.24061e-01, 8.23082e-01,
   8.22100e-01, 8.21116e-01, 8.20129e-01, 8.19140e-01, 8.18149e-01, 8.17155e-01, 8.16159e-01, 8.15160e-01, 8.14159e-01, 8.13156e-01,
   8.12150e-01, 8.11141e-01, 8.10131e-01, 8.09118e-01, 8.08102e-01, 8.07085e-01, 8.06064e-01, 8.05042e-01, 8.04017e-01, 8.02990e-01,
   8.01960e-01, 8.00928e-01, 7.99894e-01, 7.98857e-01, 7.97818e-01, 7.96777e-01, 7.95733e-01, 7.94687e-01, 7.93638e-01, 7.92588e-01,
   7.91535e-01, 7.90479e-01, 7.89421e-01, 7.88361e-01, 7.87299e-01, 7.86234e-01, 7.85167e-01, 7.84098e-01, 7.83027e-01, 7.81953e-01,
   7.80877e-01, 7.79798e-01, 7.78717e-01, 7.77634e-01, 7.76549e-01, 7.75462e-01, 7.74372e-01, 7.73280e-01, 7.72185e-01, 7.71089e-01,
   7.69990e-01, 7.68889e-01, 7.67785e-01, 7.66680e-01, 7.65572e-01, 7.64462e-01, 7.63350e-01, 7.62235e-01, 7.61118e-01, 7.59999e-01,
   7.58878e-01, 7.57755e-01, 7.56629e-01, 7.55501e-01, 7.54371e-01, 7.53239e-01, 7.52104e-01, 7.50968e-01, 7.49829e-01, 7.48688e-01,
   7.47545e-01, 7.46399e-01, 7.45252e-01, 7.44102e-01, 7.42950e-01, 7.41796e-01, 7.40640e-01, 7.39482e-01, 7.38321e-01, 7.37159e-01,
   7.35994e-01, 7.34827e-01, 7.33658e-01, 7.32487e-01, 7.31313e-01, 7.30138e-01, 7.28961e-01, 7.27781e-01, 7.26599e-01, 7.25415e-01,
   7.24229e-01, 7.23041e-01, 7.21851e-01, 7.20659e-01, 7.19465e-01, 7.18268e-01, 7.17070e-01, 7.15869e-01, 7.14667e-01, 7.13462e-01,
   7.12256e-01, 7.11047e-01, 7.09836e-01, 7.08623e-01, 7.07408e-01, 7.06191e-01, 7.04972e-01, 7.03751e-01, 7.02528e-01, 7.01303e-01,
   7.00076e-01, 6.98847e-01, 6.97616e-01, 6.96383e-01, 6.95148e-01, 6.93911e-01, 6.92672e-01, 6.91431e-01, 6.90188e-01, 6.88943e-01,
   6.87696e-01, 6.86447e-01, 6.85196e-01, 6.83943e-01, 6.82689e-01, 6.81432e-01, 6.80173e-01, 6.78913e-01, 6.77650e-01, 6.76386e-01,
   6.75119e-01, 6.73851e-01, 6.72581e-01, 6.71308e-01, 6.70034e-01, 6.68758e-01, 6.67480e-01, 6.66201e-01, 6.64919e-01, 6.63635e-01,
   6.62350e-01, 6.61063e-01, 6.59774e-01, 6.58482e-01, 6.57190e-01, 6.55895e-01, 6.54598e-01, 6.53300e-01, 6.51999e-01, 6.50697e-01,
   6.49393e-01, 6.48088e-01, 6.46780e-01, 6.45470e-01, 6.44159e-01, 6.42846e-01, 6.41531e-01, 6.40215e-01, 6.38896e-01, 6.37576e-01,
   6.36254e-01, 6.34930e-01, 6.33604e-01, 6.32277e-01, 6.30948e-01, 6.29617e-01, 6.28284e-01, 6.26950e-01, 6.25614e-01, 6.24276e-01,
   6.22936e-01, 6.21595e-01, 6.20252e-01, 6.18907e-01, 6.17561e-01, 6.16212e-01, 6.14862e-01, 6.13511e-01, 6.12157e-01, 6.10802e-01,
   6.09446e-01, 6.08087e-01, 6.06727e-01, 6.05366e-01, 6.04002e-01, 6.02637e-01, 6.01271e-01, 5.99902e-01, 5.98532e-01, 5.97161e-01,
   5.95787e-01, 5.94413e-01, 5.93036e-01, 5.91658e-01, 5.90278e-01, 5.88897e-01, 5.87514e-01, 5.86129e-01, 5.84743e-01, 5.83356e-01,
   5.81966e-01, 5.80576e-01, 5.79183e-01, 5.77789e-01, 5.76394e-01, 5.74997e-01, 5.73598e-01, 5.72198e-01, 5.70796e-01, 5.69393e-01,
   5.67988e-01, 5.66582e-01, 5.65174e-01, 5.63765e-01, 5.62354e-01, 5.60942e-01, 5.59528e-01, 5.58113e-01, 5.56696e-01, 5.55278e-01,
   5.53859e-01, 5.52437e-01, 5.51015e-01, 5.49591e-01, 5.48165e-01, 5.46739e-01, 5.45310e-01, 5.43881e-01, 5.42449e-01, 5.41017e-01,
   5.39583e-01, 5.38148e-01, 5.36711e-01, 5.35273e-01, 5.33833e-01, 5.32392e-01, 5.30950e-01, 5.29507e-01, 5.28062e-01, 5.26615e-01,
   5.25168e-01, 5.23719e-01, 5.22268e-01, 5.20817e-01, 5.19364e-01, 5.17909e-01, 5.16454e-01, 5.14997e-01, 5.13539e-01, 5.12079e-01,
   5.10619e-01, 5.09157e-01, 5.07693e-01, 5.06229e-01, 5.04763e-01, 5.03296e-01, 5.01828e-01, 5.00358e-01, 4.98887e-01, 4.97416e-01,
   4.95942e-01, 4.94468e-01, 4.92992e-01, 4.91516e-01, 4.90038e-01, 4.88559e-01, 4.87078e-01, 4.85597e-01, 4.84114e-01, 4.82630e-01,
   4.81146e-01, 4.79660e-01, 4.78172e-01, 4.76684e-01, 4.75195e-01, 4.73704e-01, 4.72213e-01, 4.70720e-01, 4.69226e-01, 4.67731e-01,
   4.66235e-01, 4.64738e-01, 4.63240e-01, 4.61741e-01, 4.60241e-01, 4.58740e-01, 4.57237e-01, 4.55734e-01, 4.54230e-01, 4.52725e-01,
   4.51219e-01, 4.49711e-01, 4.48203e-01, 4.46694e-01, 4.45184e-01, 4.43673e-01, 4.42161e-01, 4.40648e-01, 4.39134e-01, 4.37619e-01,
   4.36103e-01, 4.34587e-01, 4.33069e-01, 4.31551e-01, 4.30031e-01, 4.28511e-01, 4.26990e-01, 4.25468e-01, 4.23945e-01, 4.22422e-01,
   4.20897e-01, 4.19372e-01, 4.17846e-01, 4.16319e-01, 4.14791e-01, 4.13263e-01, 4.11734e-01, 4.10204e-01, 4.08673e-01, 4.07142e-01,
   4.05609e-01, 4.04076e-01, 4.02543e-01, 4.01008e-01, 3.99473e-01, 3.97937e-01, 3.96401e-01, 3.94864e-01, 3.93326e-01, 3.91787e-01,
   3.90248e-01, 3.88708e-01, 3.87168e-01, 3.85627e-01, 3.84085e-01, 3.82543e-01, 3.81000e-01, 3.79457e-01, 3.77913e-01, 3.76369e-01,
   3.74824e-01, 3.73278e-01, 3.71732e-01, 3.70185e-01, 3.68638e-01, 3.67091e-01, 3.65542e-01, 3.63994e-01, 3.62445e-01, 3.60896e-01,
   3.59346e-01, 3.57795e-01, 3.56245e-01, 3.54694e-01, 3.53142e-01, 3.51590e-01, 3.50038e-01, 3.48485e-01, 3.46932e-01, 3.45379e-01,
   3.43826e-01, 3.42272e-01, 3.40718e-01, 3.39163e-01, 3.37608e-01, 3.36053e-01, 3.34498e-01, 3.32943e-01, 3.31387e-01, 3.29831e-01,
   3.28275e-01, 3.26719e-01, 3.25163e-01, 3.23606e-01, 3.22050e-01, 3.20493e-01, 3.18936e-01, 3.17379e-01, 3.15822e-01, 3.14265e-01,
   3.12708e-01, 3.11150e-01, 3.09593e-01, 3.08036e-01, 3.06479e-01, 3.04922e-01, 3.03365e-01, 3.01808e-01, 3.00251e-01, 2.98694e-01,
   2.97137e-01, 2.95580e-01, 2.94024e-01, 2.92467e-01, 2.90911e-01, 2.89355e-01, 2.87799e-01, 2.86244e-01, 2.84689e-01, 2.83134e-01,
   2.81579e-01, 2.80024e-01, 2.78470e-01, 2.76916e-01, 2.75363e-01, 2.73810e-01, 2.72257e-01, 2.70705e-01, 2.69153e-01, 2.67601e-01,
   2.66050e-01, 2.64500e-01, 2.62950e-01, 2.61400e-01, 2.59851e-01, 2.58303e-01, 2.56755e-01, 2.55208e-01, 2.53661e-01, 2.52116e-01,
   2.50570e-01, 2.49026e-01, 2.47482e-01, 2.45939e-01, 2.44397e-01, 2.42855e-01, 2.41315e-01, 2.39775e-01, 2.38236e-01, 2.36698e-01,
   2.35161e-01, 2.33624e-01, 2.32089e-01, 2.30555e-01, 2.29022e-01, 2.27489e-01, 2.25958e-01, 2.24428e-01, 2.22899e-01, 2.21372e-01,
   2.19845e-01, 2.18320e-01, 2.16795e-01, 2.15273e-01, 2.13751e-01, 2.12231e-01, 2.10712e-01, 2.09194e-01, 2.07678e-01, 2.06163e-01,
   2.04650e-01, 2.03139e-01, 2.01628e-01, 2.00120e-01, 1.98613e-01, 1.97108e-01, 1.95604e-01, 1.94102e-01, 1.92602e-01, 1.91104e-01,
   1.89607e-01, 1.88112e-01, 1.86619e-01, 1.85129e-01, 1.83640e-01, 1.82153e-01, 1.80668e-01, 1.79185e-01, 1.77704e-01, 1.76226e-01,
   1.74750e-01, 1.73276e-01, 1.71804e-01, 1.70334e-01, 1.68867e-01, 1.67403e-01, 1.65941e-01, 1.64481e-01, 1.63024e-01, 1.61569e-01,
   1.60118e-01, 1.58668e-01, 1.57222e-01, 1.55778e-01, 1.54338e-01, 1.52900e-01, 1.51465e-01, 1.50033e-01, 1.48604e-01, 1.47178e-01,
   1.45755e-01, 1.44336e-01, 1.42919e-01, 1.41506e-01, 1.40097e-01, 1.38691e-01, 1.37288e-01, 1.35889e-01, 1.34493e-01, 1.33101e-01,
   1.31713e-01, 1.30328e-01, 1.28948e-01, 1.27571e-01, 1.26198e-01, 1.24829e-01, 1.23465e-01, 1.22104e-01, 1.20748e-01, 1.19396e-01,
   1.18048e-01, 1.16704e-01, 1.15365e-01, 1.14031e-01, 1.12701e-01, 1.11376e-01, 1.10056e-01, 1.08740e-01, 1.07430e-01, 1.06124e-01,
   1.04823e-01, 1.03528e-01, 1.02237e-01, 1.00952e-01, 9.96726e-02, 9.83982e-02, 9.71294e-02, 9.58661e-02, 9.46084e-02, 9.33564e-02,
   9.21103e-02, 9.08700e-02, 8.96357e-02, 8.84075e-02, 8.71853e-02, 8.59694e-02, 8.47597e-02, 8.35565e-02, 8.23596e-02, 8.11693e-02,
   7.99857e-02, 7.88087e-02, 7.76385e-02, 7.64752e-02, 7.53189e-02, 7.41695e-02, 7.30274e-02, 7.18924e-02, 7.07647e-02, 6.96444e-02,
   6.85316e-02, 6.74263e-02, 6.63287e-02, 6.52388e-02, 6.41568e-02, 6.30826e-02, 6.20164e-02, 6.09582e-02, 5.99083e-02, 5.88665e-02,
   5.78331e-02, 5.68081e-02, 5.57915e-02, 5.47836e-02, 5.37842e-02, 5.27936e-02, 5.18118e-02, 5.08389e-02, 4.98749e-02, 4.89200e-02,
   4.79742e-02, 4.70376e-02, 4.61102e-02, 4.51922e-02, 4.42836e-02, 4.33844e-02, 4.24948e-02, 4.16148e-02, 4.07445e-02, 3.98839e-02,
   3.90330e-02, 3.81920e-02, 3.73610e-02, 3.65398e-02, 3.57287e-02, 3.49277e-02, 3.41367e-02, 3.33559e-02, 3.25853e-02, 3.18250e-02,
   3.10749e-02, 3.03351e-02, 2.96056e-02, 2.88865e-02, 2.81778e-02, 2.74795e-02, 2.67917e-02, 2.61143e-02, 2.54473e-02, 2.47909e-02,
   2.41449e-02, 2.35094e-02, 2.28844e-02, 2.22698e-02, 2.16658e-02, 2.10721e-02, 2.04890e-02, 1.99163e-02, 1.93539e-02, 1.88020e-02,
   1.82604e-02, 1.77292e-02, 1.72082e-02, 1.66975e-02, 1.61970e-02, 1.57066e-02, 1.52263e-02, 1.47561e-02, 1.42959e-02, 1.38456e-02,
   1.34051e-02, 1.29745e-02, 1.25536e-02, 1.21423e-02, 1.17405e-02, 1.13483e-02, 1.09654e-02, 1.05919e-02, 1.02275e-02, 9.87231e-03,
   9.52609e-03, 9.18877e-03, 8.86026e-03, 8.54045e-03, 8.22922e-03, 7.92647e-03, 7.63207e-03, 7.34591e-03, 7.06786e-03, 6.79782e-03,
   6.53564e-03, 6.28121e-03, 6.03439e-03, 5.79506e-03, 5.56309e-03, 5.33834e-03, 5.12068e-03, 4.90998e-03, 4.70609e-03, 4.50889e-03,
   4.31824e-03, 4.13400e-03, 3.95602e-03, 3.78418e-03, 3.61834e-03, 3.45835e-03, 3.30408e-03, 3.15538e-03, 3.01214e-03, 2.87420e-03,
   2.74142e-03, 2.61369e-03, 2.49085e-03, 2.37278e-03, 2.25934e-03, 2.15041e-03, 2.04585e-03, 1.94554e-03, 1.84934e-03, 1.75713e-03,
   1.66879e-03, 1.58420e-03, 1.50323e-03, 1.42578e-03, 1.35171e-03, 1.28092e-03, 1.21329e-03, 1.14872e-03, 1.08710e-03, 1.02832e-03,
   9.72276e-04, 9.18868e-04, 8.67998e-04, 8.19568e-04, 7.73485e-04, 7.29657e-04, 6.87995e-04, 6.48410e-04, 6.10819e-04, 5.75138e-04,
   5.41288e-04, 5.09191e-04, 4.78772e-04, 4.49958e-04, 4.22677e-04, 3.96862e-04, 3.72446e-04, 3.49365e-04, 3.27557e-04, 3.06963e-04,
   2.87525e-04, 2.69188e-04, 2.51898e-04, 2.35604e-04, 2.20256e-04, 2.05808e-04, 1.92213e-04, 1.79428e-04, 1.67411e-04, 1.56122e-04,
   1.45521e-04, 1.35573e-04, 1.26243e-04, 1.17495e-04, 1.09299e-04, 1.01624e-04, 9.44401e-05, 8.77197e-05, 8.14363e-05, 7.55646e-05,
   7.00806e-05, 6.49614e-05, 6.01853e-05, 5.57318e-05, 5.15813e-05, 4.77152e-05, 4.41161e-05, 4.07673e-05, 3.76532e-05, 3.47588e-05,
   3.20701e-05, 2.95738e-05, 2.72576e-05, 2.51095e-05, 2.31184e-05, 2.12740e-05, 1.95663e-05, 1.79861e-05, 1.65247e-05, 1.51739e-05,
   1.39260e-05, 1.27738e-05, 1.17107e-05, 1.07302e-05, 9.82645e-06, 8.99392e-06, 8.22742e-06, 7.52213e-06, 6.87351e-06, 6.27736e-06,
   5.72974e-06, 5.22700e-06, 4.76572e-06, 4.34272e-06, 3.95505e-06, 3.59996e-06, 3.27491e-06, 2.97753e-06, 2.70562e-06, 2.45715e-06,
   2.23022e-06, 2.02309e-06, 1.83415e-06, 1.66190e-06, 1.50496e-06, 1.36205e-06, 1.23199e-06, 1.11370e-06, 1.00619e-06, 9.08519e-07,
   8.19848e-07, 7.39395e-07, 6.66443e-07, 6.00333e-07, 5.40459e-07, 4.86265e-07, 4.37244e-07, 3.92929e-07, 3.52892e-07, 3.16743e-07,
   2.84124e-07, 2.54710e-07, 2.28201e-07, 2.04325e-07, 1.82836e-07, 1.63505e-07, 1.46128e-07, 1.30516e-07, 1.16500e-07, 1.03924e-07,
   9.26468e-08, 8.25417e-08, 7.34923e-08, 6.53936e-08, 5.81503e-08, 5.16763e-08, 4.58936e-08, 4.07318e-08, 3.61273e-08, 3.20225e-08,
   2.83657e-08, 2.51101e-08, 2.22136e-08, 1.96383e-08, 1.73501e-08, 1.53184e-08, 1.35157e-08, 1.19171e-08, 1.05006e-08, 9.24625e-09,
   8.13625e-09, 7.15467e-09, 6.28724e-09, 5.52121e-09, 4.84520e-09, 4.24904e-09, 3.72366e-09, 3.26099e-09, 2.85383e-09, 2.49576e-09,
   2.18110e-09, 1.90477e-09, 1.66227e-09, 1.44962e-09, 1.26328e-09, 1.10011e-09, 9.57322e-10, 8.32472e-10, 7.23383e-10, 6.28134e-10,
   5.45029e-10, 4.72575e-10, 4.09452e-10, 3.54500e-10, 3.06696e-10, 2.65142e-10, 2.29047e-10, 1.97718e-10, 1.70546e-10, 1.46997e-10,
   1.26604e-10, 1.08958e-10, 9.36992e-11, 8.05157e-11, 6.91340e-11, 5.93153e-11, 5.08516e-11, 4.35617e-11, 3.72876e-11, 3.18921e-11,
   2.72559e-11
};

// Drop-in replacement for cosine function for describing plasma production at the height of max plasma production
// using the Chapman function (which assumes the earth is round, not flat).
//
// The advantage of this approach is that the conductance gradient at the terminator is
// more realistic. This is important since conductance gradients appear in the equations that
// relate electric and magnetic fields. In addition, conductances above 90° sza are positive.
// The code is based on table lookup, and does not calculate the Chapman function.
// Author: S. M. Hatch (2024)
Real altcos(Real sza) {
   Real degrees = fabs(sza) / M_PI * 180;

   // Clamp to table lookup range
   degrees = max(0.,degrees);
   degrees = min(120.,degrees);

   int bin = degrees * 10.;
   Real interpolant = bin - (degrees * 10.);
   return (1.-interpolant) * chapman_euv_table[bin] + interpolant * chapman_euv_table[bin+1];
}

void assignConductivityTensor(std::vector<SphericalTriGrid::Node>& nodes, Real sigmaP, Real sigmaH) {
   static const char epsilon[3][3][3] = {
      {{0,0,0},{0,0,1},{0,-1,0}},
      {{0,0,-1},{0,0,0},{1,0,0}},
      {{0,1,0},{-1,0,0},{0,0,0}}
   };

   for(uint n=0; n<nodes.size(); n++) {
      std::array<Real, 3> b = {nodes[n].x[0] / Ionosphere::innerRadius, nodes[n].x[1] / Ionosphere::innerRadius, nodes[n].x[2] / Ionosphere::innerRadius};
      if(nodes[n].x[2] >= 0) {
         b[0] *= -1;
         b[1] *= -1;
         b[2] *= -1;
      }
      for(int i=0; i<3; i++) {
         for(int j=0; j<3; j++) {
            nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] = sigmaP * (((i==j)? 1. : 0.) - b[i]*b[j]);
            for(int k=0; k<3; k++) {
               nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] -= sigmaH * epsilon[i][j][k]*b[k];
            }
         }
      }
   }
}

void assignConductivityTensorFromLoadedData(std::vector<SphericalTriGrid::Node>& nodes) {
   static const char epsilon[3][3][3] = {
      {{0,0,0},{0,0,1},{0,-1,0}},
      {{0,0,-1},{0,0,0},{1,0,0}},
      {{0,1,0},{-1,0,0},{0,0,0}}
   };

   for(uint n=0; n<nodes.size(); n++) {
      Real sigmaH = nodes[n].parameters[ionosphereParameters::SIGMAH];
      Real sigmaP = nodes[n].parameters[ionosphereParameters::SIGMAP];
      std::array<Real, 3> b = {nodes[n].x[0] / Ionosphere::innerRadius, nodes[n].x[1] / Ionosphere::innerRadius, nodes[n].x[2] / Ionosphere::innerRadius};
      if(nodes[n].x[2] >= 0) {
         b[0] *= -1;
         b[1] *= -1;
         b[2] *= -1;
      }
      for(int i=0; i<3; i++) {
         for(int j=0; j<3; j++) {
            nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] = sigmaP * (((i==j)? 1. : 0.) - b[i]*b[j]);
            for(int k=0; k<3; k++) {
               nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] -= sigmaH * epsilon[i][j][k]*b[k];
            }
         }
      }
   }
}

std::vector<Real> edgeJCurl;
std::vector<Real> edgeJDiv;
std::vector<Real> edgeLength;
OpenBucketHashtable<uint64_t, uint> edgeIndex;

// Unique lookup of edges given a pair of nodes.
std::tuple<uint,int> getEdgeIndexOrientation(uint32_t a, uint32_t b)  {

   int orientation = 0;

   // Edges are sorted by adjacent node index (directed to go from smaller to larger index)
   uint32_t low = std::min(a,b);
   uint32_t high = std::max(a,b);

   // If a->b is the natural ordering of this edge, return 1 for orientation, otherwise -1
   if(low == a) {
      orientation = 1;
   } else {
      orientation = -1;
   }

   // We use a 64bit value of both edges as the hash value
   uint64_t hash = high;
   hash <<= 32;
   hash |= low;

   if(edgeIndex.count(hash) == 0) {
      // Add entry into array
      edgeIndex[hash] = edgeJCurl.size();
      edgeJCurl.push_back(0.);
      edgeJDiv.push_back(0.);
      edgeLength.push_back(0.);
   }

   return {edgeIndex[hash], orientation};
}

// Interpolate edge-based quantity to elements (barycentres) using Whitney 1-forms.
// (DOI: 10.1145/1141911.1141991)
Eigen::Vector3d whitneyInterpolate(SphericalTriGrid& grid, uint32_t el, std::vector<Real> edgeValue) {
   std::array<uint32_t, 3>& corners = grid.elements[el].corners;
   Real A = grid.elementArea(el);

   auto [e1,o1] = getEdgeIndexOrientation(corners[0],corners[1]);
   auto [e2,o2] = getEdgeIndexOrientation(corners[1],corners[2]);
   auto [e3,o3] = getEdgeIndexOrientation(corners[2],corners[0]);

   Eigen::Vector3d r0(grid.nodes[corners[0]].x.data());
   Eigen::Vector3d r1(grid.nodes[corners[1]].x.data());
   Eigen::Vector3d r2(grid.nodes[corners[2]].x.data());

   Eigen::Vector3d barycentre = (r0+r1+r2)/3.;

   // Barycentric coordinates
   auto lambda1 = [&r0,&r1,&r2,&A](const Eigen::Vector3d& p) {
      return ((r0-p).cross(r1-p)).norm() / (2*A);
   };
   auto lambda2 = [&r0,&r1,&r2,&A](const Eigen::Vector3d& p) {
      return ((r1-p).cross(r2-p)).norm() / (2*A);
   };
   auto lambda3 = [&r0,&r1,&r2,&A](const Eigen::Vector3d& p) {
      return ((r2-p).cross(r0-p)).norm() / (2*A);
   };

   // Barycentric gradients (these are constant per element)
   Eigen::Vector3d gradLambda1 = edgeLength[e1] / (2 * A) * (r1-r0).cross(r2-r0).cross(r1-r0).normalized();
   Eigen::Vector3d gradLambda2 = edgeLength[e2] / (2 * A) * (r2-r1).cross(r0-r1).cross(r2-r1).normalized();
   Eigen::Vector3d gradLambda3 = edgeLength[e3] / (2 * A) * (r2-r0).cross(r1-r0).cross(r2-r0).normalized();

   // Whitney 1-form basis functions
   auto w1 = [&](const Eigen::Vector3d& p) {
      return lambda2(p) * gradLambda3 - lambda3(p) * gradLambda2;
   };
   auto w2 = [&](const Eigen::Vector3d& p) {
      return lambda3(p) * gradLambda1 - lambda1(p) * gradLambda3;
   };
   auto w3 = [&](const Eigen::Vector3d& p) {
      return lambda1(p) * gradLambda2 - lambda2(p) * gradLambda1;
   };

   // Effective interpolated value this element
   return o1*edgeLength[e1]*edgeValue[e1] * w1(barycentre) + o2*edgeLength[e2]*edgeValue[e2] * w2(barycentre) + o3*edgeLength[e3]*edgeValue[e3] *w3(barycentre);
}

int main(int argc, char** argv) {

   // Init MPI
   int required=MPI_THREAD_FUNNELED;
   int provided;
   int myRank;
   MPI_Init_thread(&argc,&argv,required,&provided);
   if (required > provided){
      MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
      if(myRank==MASTER_RANK)
         cerr << "(MAIN): MPI_Init_thread failed! Got " << provided << ", need "<<required <<endl;
      exit(1);
   }
   const int masterProcessID = 0;
   logFile.open(MPI_COMM_WORLD, masterProcessID, "logfile.txt");


   // Parse parameters
   int numNodes = 64;
   std::string sigmaString="identity";
   std::string facString="constant";
   std::string gaugeFixString="pole";
   std::string inputFile;
   std::string outputFilename("output.vlsv");
   std::vector<std::pair<double, double>> refineExtents;
   Ionosphere::solverMaxIterations = 1000;
   bool doPrecondition = true;
   bool writeSolverMtarix = false;
   bool quiet = false;
   bool runCurlJSolver = false;
   int multipoleL = 0;
   int multipolem = 0;
   if(argc ==1) {
      cerr << "Running with default options. Run main --help to see available settings." << endl;
   }
   for(int i=1; i<argc; i++) {
      if(!strcmp(argv[i], "-N")) {
         numNodes = atoi(argv[++i]);
         continue;
      }
      if(!strcmp(argv[i], "-r")) {
         double minLat = atof(argv[++i]);
         double maxLat = atof(argv[++i]);
         refineExtents.push_back(std::pair<double,double>(minLat, maxLat));
         continue;
      }
      if(!strcmp(argv[i], "-sigma")) {
         sigmaString = argv[++i];
         continue;
      }
      if(!strcmp(argv[i], "-fac")) {
         facString = argv[++i];

         // Special handling for multipoles
         if(facString == "multipole") {
            multipoleL = atoi(argv[++i]);
            multipolem = atoi(argv[++i]);
         }
         continue;
      }
      if(!strcmp(argv[i], "-gaugeFix")) {
         gaugeFixString = argv[++i];
         continue;
      }
      if(!strcmp(argv[i], "-np")) {
         doPrecondition = false;
         continue;
      }
      if(!strcmp(argv[i], "-infile")) {
         inputFile = argv[++i];
         continue;
      }
      if(!strcmp(argv[i], "-maxIter")) {
         Ionosphere::solverMaxIterations = atoi(argv[++i]);
         continue;
      }
      if(!strcmp(argv[i], "-o")) {
         outputFilename = argv[++i];
         continue;
      }
      if(!strcmp(argv[i], "-matrix")) {
         writeSolverMtarix = true;
         continue;
      }
      if(!strcmp(argv[i], "-q")) {
         quiet = true;
         continue;
      }
      cerr << "Unknown command line option \"" << argv[i] << "\"" << endl;
      cerr << endl;
      cerr << "main [-N num] [-r <lat0> <lat1>] [-sigma (identity|random|35|53|curlJ|file)] [-fac (constant|dipole|quadrupole|octopole|hexadecapole||file)] [-facfile <filename>] [-gaugeFix equator|equator40|equator45|equator50|equator60|pole|integral|none] [-np]" << endl;
      cerr << "Paramters:" << endl;
      cerr << " -N:            Number of ionosphere mesh nodes (default: 64)" << endl;
      cerr << " -r:            Refine grid between the given latitudes (can be specified multiple times)" << endl;
      cerr << " -sigma:        Conductivity matrix contents (default: identity)" << endl;
      cerr << "                options are:" << endl;
      cerr << "                identity - identity matrix w/ conductivity 1" << endl;
      cerr << "                ponly    - Constant pedersen conductivitu"<< endl;
      cerr << "                10 -       Sigma_H = 0, Sigma_P = 10" << endl;
      cerr << "                35 -       Sigma_H = 3, Sigma_P = 5" << endl;
      cerr << "                53 -       Sigma_H = 5, Sigma_P = 3" << endl;
      cerr << "                100 -      Sigma_H = 100, Sigma_P=20" << endl;
      cerr << "                file -     Read from vlsv input file " << endl;
      cerr << " -fac:          FAC pattern on the sphere (default: constant)" << endl;
      cerr << "                options are:" << endl;
      cerr << "                constant          - Constant value of 1" << endl;
      cerr << "                dipole            - north/south dipole" << endl;
      cerr << "                quadrupole        - east/west quadrupole (L=2, m=1)" << endl;
      cerr << "                octopole          - octopole (L=3, m=2)" << endl;
      cerr << "                hexadecapole      - hexadecapole (L=4, m=3)" << endl;
      cerr << "                multipole <L> <m> - generic multipole, L and m given separately." << endl;
      cerr << "                merkin2010        - eq13 of Merkin et al (2010)" << endl;
      cerr << "                file              - read FAC distribution from vlsv input file" << endl;
      cerr << "                pole              - testcase: FACs are nonzero only at the north pole" << endl;
      cerr << " -infile:       Read FACs from this input file" << endl;
      cerr << " -gaugeFix:     Solver gauge fixing method (default: pole)" << endl;
      cerr << "                options are:" << endl;
      cerr << "                pole      - Fix potential in a single node at the north pole" << endl;
      cerr << "                equator   - Fix potential on all nodes +- 10 degrees of the equator" << endl;
      cerr << "                equator40 - Fix potential on all nodes +- 40 degrees of the equator" << endl;
      cerr << "                equator45 - Fix potential on all nodes +- 45 degrees of the equator" << endl;
      cerr << "                equator50 - Fix potential on all nodes +- 50 degrees of the equator" << endl;
      cerr << "                equator60 - Fix potential on all nodes +- 60 degrees of the equator" << endl;
      cerr << " -np:           DON'T use the matrix preconditioner (default: do)" << endl;
      cerr << " -maxIter:      Maximum number of solver iterations" << endl;
      cerr << " -o <filename>: Output filename (default: \"output.vlsv\")" << endl;
      cerr << " -matrix:       Write solver dependency matrix to solverMatrix.txt (default: don't.)" << endl;
      cerr << " -q:            Quiet mode (only output residual value" << endl;

      return 1;
   }

   phiprof::initialize();

   // Initialize ionosphere grid
   Ionosphere::innerRadius =  physicalconstants::R_E + 100e3;
   if(numNodes > 0) {
      ionosphereGrid.initializeSphericalFibonacci(numNodes);
   } else {
      ionosphereGrid.initializeOctahedron();
   }
   if(gaugeFixString == "pole") {
      ionosphereGrid.gaugeFixing = SphericalTriGrid::Pole;
   } else if (gaugeFixString == "integral") {
      ionosphereGrid.gaugeFixing = SphericalTriGrid::Integral;
   } else if (gaugeFixString == "equator") {
      ionosphereGrid.gaugeFixing = SphericalTriGrid::Equator;
      Ionosphere::shieldingLatitude = 10.;
   } else if (gaugeFixString == "equator40") {
      ionosphereGrid.gaugeFixing = SphericalTriGrid::Equator;
      Ionosphere::shieldingLatitude = 40.;
   } else if (gaugeFixString == "equator45") {
      ionosphereGrid.gaugeFixing = SphericalTriGrid::Equator;
      Ionosphere::shieldingLatitude = 45.;
   } else if (gaugeFixString == "equator50") {
      ionosphereGrid.gaugeFixing = SphericalTriGrid::Equator;
      Ionosphere::shieldingLatitude = 50.;
   } else if (gaugeFixString == "equator60") {
      ionosphereGrid.gaugeFixing = SphericalTriGrid::Equator;
      Ionosphere::shieldingLatitude = 60.;
   } else if (gaugeFixString == "none") {
      ionosphereGrid.gaugeFixing = SphericalTriGrid::None;
   } else {
      cerr << "Unknown gauge fixing method " << gaugeFixString << endl;
      return 1;
   }

   // Refine the base shape to acheive desired resolution
   auto refineBetweenLatitudes = [](Real phi1, Real phi2) -> void {
      uint numElems=ionosphereGrid.elements.size();

      for(uint i=0; i< numElems; i++) {
         Real mean_z = 0;
         mean_z  = ionosphereGrid.nodes[ionosphereGrid.elements[i].corners[0]].x[2];
         mean_z += ionosphereGrid.nodes[ionosphereGrid.elements[i].corners[1]].x[2];
         mean_z += ionosphereGrid.nodes[ionosphereGrid.elements[i].corners[2]].x[2];
         mean_z /= 3.;

         if(fabs(mean_z) >= sin(phi1 * M_PI / 180.) * Ionosphere::innerRadius &&
               fabs(mean_z) <= sin(phi2 * M_PI / 180.) * Ionosphere::innerRadius) {
            ionosphereGrid.subdivideElement(i);
         }
      }
   };

   if(refineExtents.size() > 0) {
      for(unsigned int i=0; i< refineExtents.size(); i++) {
         refineBetweenLatitudes(refineExtents[i].first, refineExtents[i].second);
      }
      ionosphereGrid.stitchRefinementInterfaces();
   }


   std::vector<SphericalTriGrid::Node>& nodes = ionosphereGrid.nodes;
   std::vector< Real > elementCorrectionFactors(ionosphereGrid.elements.size());
   std::vector< Eigen::Vector3d > elementCurlFreeCurrent(ionosphereGrid.elements.size());
   std::vector< Eigen::Vector3d > elementDivFreeCurrent(ionosphereGrid.elements.size());

   // Set FACs
   if(facString == "constant") {
      for(uint n=0; n<nodes.size(); n++) {
         nodes[n].parameters[ionosphereParameters::SOURCE] = 1;
      }
   } else if(facString == "dipole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = getDualPolygonArea(ionosphereGrid, n);
         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(1,0,theta) * cos(0*phi) * area;
      }
   } else if(facString == "quadrupole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = getDualPolygonArea(ionosphereGrid, n);
         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(2,1,theta) * cos(1*phi) * area;
      }
   } else if(facString == "octopole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = getDualPolygonArea(ionosphereGrid, n);
         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(3,2,theta) * cos(2*phi) * area;
      }
   } else if(facString == "hexadecapole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = getDualPolygonArea(ionosphereGrid, n);
         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(4,3,theta) * cos(3*phi) * area;
      }
   } else if(facString == "multipole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = getDualPolygonArea(ionosphereGrid, n);
         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(multipoleL,fabs(multipolem),theta) * cos(multipolem*phi) * area;
      }
   } else if(facString == "merkin2010") {

      // From Merkin et al (2010), LFM's conductivity map test setup (Figure3 / eq 13):
      const double j_0 = 1e-6;
      const double theta_0 = 22. / 180 * M_PI;
      const double deltaTheta = 12. / 180 * M_PI;

      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = getDualPolygonArea(ionosphereGrid, n);
         double j_parallel=0;

         // Merkin et al specifies colatitude as degrees-from-the-pole
         if(fabs(theta) >= theta_0 && fabs(theta) < theta_0 + deltaTheta) {
            j_parallel = j_0 * sin(M_PI/2 - fabs(theta)) * sin(phi);
         }
         nodes[n].parameters[ionosphereParameters::SOURCE] = j_parallel * area;
      }
   } else if(facString == "file") {
      vlsv::ParallelReader inVlsv;
      inVlsv.open(inputFile,MPI_COMM_WORLD,masterProcessID);
      readIonosphereNodeVariable(inVlsv, "ig_fac", ionosphereGrid, ionosphereParameters::SOURCE);
      for(uint i=0; i<ionosphereGrid.nodes.size(); i++) {
         Real area = 0;
         for(uint e=0; e<ionosphereGrid.nodes[i].numTouchingElements; e++) {
            area += ionosphereGrid.elementArea(ionosphereGrid.nodes[i].touchingElements[e]);
         }
         area /= 3.; // As every element has 3 corners, don't double-count areas
         ionosphereGrid.nodes[i].parameters[ionosphereParameters::SOURCE] *= area;
      }
      // Also read open/closed information from the file, if it exists.
      // (We use PPARAM as temporary storage here)
      readIonosphereNodeVariable(inVlsv, "ig_openclosed", ionosphereGrid, ionosphereParameters::PPARAM);
      for(uint i=0; i<ionosphereGrid.nodes.size(); i++) {
         ionosphereGrid.nodes[i].openFieldLine = ionosphereGrid.nodes[i].parameters[ionosphereParameters::PPARAM];
      }
   } else if(facString == "pole") {
      for(uint i=0; i<ionosphereGrid.nodes.size(); i++) {
         if(ionosphereGrid.nodes[i].x[2] >= Ionosphere::innerRadius * 0.95) {
            ionosphereGrid.nodes[i].parameters[ionosphereParameters::SOURCE] = 1;
         } else if(ionosphereGrid.nodes[i].x[2] <= -Ionosphere::innerRadius * 0.95) {
            ionosphereGrid.nodes[i].parameters[ionosphereParameters::SOURCE] = -1;
         } else {
            ionosphereGrid.nodes[i].parameters[ionosphereParameters::SOURCE] = 0;
         }
      }
   } else {
      cerr << "FAC pattern " << sigmaString << " not implemented!" << endl;
      return 1;
   }


   // Set conductivity tensors
   if(sigmaString == "identity") {
      for(uint n=0; n<nodes.size(); n++) {
         for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
               nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] = ((i==j)? 1. : 0.);
            }
         }
      }
   } else if(sigmaString == "file") {
      vlsv::ParallelReader inVlsv;
      inVlsv.open(inputFile,MPI_COMM_WORLD,masterProcessID);
      // Try to read the sigma tensor directly
      if(!readIonosphereNodeVariable(inVlsv, "ig_sigma", ionosphereGrid, ionosphereParameters::SIGMA)) {

         // If that doesn't exist, reconstruct it from the sigmaH and sigmaP components
         // (This assumes that the input file was run with the "GUMICS" conductivity model. Which is reasonable,
         // because the others don't work very well)
         if(!quiet) {
            cerr << "Reading conductivity tensor from ig_sigmah, ig_sigmap." << endl;
         }
         readIonosphereNodeVariable(inVlsv, "ig_sigmah", ionosphereGrid, ionosphereParameters::SIGMAH);
         readIonosphereNodeVariable(inVlsv, "ig_sigmap", ionosphereGrid, ionosphereParameters::SIGMAP);
         //readIonosphereNodeVariable(inVlsv, "ig_sigmaparallel", ionosphereGrid, ionosphereParameters::SIGMAPARALLEL);
         assignConductivityTensorFromLoadedData(nodes);
      }
   } else if(sigmaString == "ponly") {
         Real sigmaP=3.;
         Real sigmaH=0.;
         assignConductivityTensor(nodes, sigmaP, sigmaH);
   } else if(sigmaString == "10") {
         Real sigmaP=10.;
         Real sigmaH=0.;
         assignConductivityTensor(nodes, sigmaP, sigmaH);
   } else if(sigmaString == "35") {
         Real sigmaP=3.;
         Real sigmaH=5.;
         assignConductivityTensor(nodes, sigmaP, sigmaH);
   } else if(sigmaString == "53") {
         Real sigmaP=5.;
         Real sigmaH=3.;
         assignConductivityTensor(nodes, sigmaP, sigmaH);
   } else if(sigmaString == "10") {
         Real sigmaP=10.;
         Real sigmaH=0.;
         assignConductivityTensor(nodes, sigmaP, sigmaH);
   } else if(sigmaString == "100") {
         Real sigmaP=20.;
         Real sigmaH=100.;
         assignConductivityTensor(nodes, sigmaP, sigmaH);
   } else if(sigmaString == "curlJ") {
      runCurlJSolver = true;

      // First, solve curl-free inplane current system.
      // Use those currents to estimate sigma ratio.
      // Then, solve divergence-free part.
      // Ginally, estimate Sigmas.

      // Fill edge arrays
      for(uint32_t el=0; el< ionosphereGrid.elements.size(); el++) {
         SphericalTriGrid::Element& element = ionosphereGrid.elements[el];

         int i=element.corners[0],j=element.corners[1],k=element.corners[2];
         Eigen::Vector3d r0(nodes[i].x.data());
         Eigen::Vector3d r1(nodes[j].x.data());
         Eigen::Vector3d r2(nodes[k].x.data());

         // Edge length
         auto [e,orientation] = getEdgeIndexOrientation(i,j);
         edgeLength[e] = (r0-r1).norm();

         std::tie(e, orientation) = getEdgeIndexOrientation(j,k);
         edgeLength[e] = (r1-r2).norm();

         std::tie(e, orientation) = getEdgeIndexOrientation(k,i);
         edgeLength[e] = (r0-r2).norm();
      }

      // Eigen vector and matrix for solving
      Eigen::VectorXd vJ(edgeJCurl.size());
      Eigen::VectorXd vRHS1(nodes.size() + NODE_CONSTRAINT_REDUCTION + ionosphereGrid.elements.size() + ELEMENT_CONSTRAINT_REDUCTION); // Right hand side for divergence-free system
      Eigen::VectorXd vRHS2(nodes.size() + NODE_CONSTRAINT_REDUCTION + ionosphereGrid.elements.size() + ELEMENT_CONSTRAINT_REDUCTION); // Right hand side for curl-free system
      Eigen::SparseMatrix<Real> curlSolverMatrix(nodes.size() + NODE_CONSTRAINT_REDUCTION + ionosphereGrid.elements.size() + ELEMENT_CONSTRAINT_REDUCTION, edgeJCurl.size());

      // Divergence constraints
      for(uint m=0; m<nodes.size() + NODE_CONSTRAINT_REDUCTION; m++) {

         // Divergence

         // Calculate the effective area of the Voronoi cell surrounding this node
         Real dualPolygonArea = getDualPolygonArea(ionosphereGrid, m);

         vRHS1[m] = 0;
         vRHS2[m] = ionosphereGrid.nodes[m].parameters[ionosphereParameters::SOURCE];

         for(uint32_t el=0; el< nodes[m].numTouchingElements; el++) {
            int32_t elementm = nodes[m].touchingElements[el];
            SphericalTriGrid::Element& element = ionosphereGrid.elements[elementm];

            // Find the two other nodes on this element
            int i=0,j=0;
            int cm=0,ci=0,cj=0;
            for(int c=0; c< 3; c++) {
               if(element.corners[c] == m) {
                  cm = c;
                  ci = (c+1)%3;
                  i=element.corners[ci];
                  cj = (c+2)%3;
                  j=element.corners[cj];
                  break;
               }
            }
            Eigen::Vector3d ri(nodes[i].x.data());
            Eigen::Vector3d rj(nodes[j].x.data());
            Eigen::Vector3d rm(nodes[m].x.data());

            int32_t otherElementi = ionosphereGrid.findElementNeighbour(elementm, cm, ci);
            int32_t otherElementj = ionosphereGrid.findElementNeighbour(elementm, cm, cj);

            auto [barycentrei,intersectioni] = getConnectingSegmentLengths(ionosphereGrid, elementm, otherElementi);
            auto [barycentrej,intersectionj] = getConnectingSegmentLengths(ionosphereGrid, elementm, otherElementj);

            Eigen::Vector3d barycentrem = getElementBarycentre(ionosphereGrid, elementm);

            Real gaussIntegralContributioni = ((intersectioni - barycentrem).cross( (ri-rm).normalized())).norm()
                  + ((intersectioni - barycentrei).cross( (ri-rm).normalized())).norm();
            Real gaussIntegralContributionj = ((intersectionj - barycentrem).cross( (rj-rm).normalized())).norm()
                  + ((intersectionj - barycentrej).cross( (rj-rm).normalized())).norm();

            auto [e,orientation] = getEdgeIndexOrientation(m,i);
            curlSolverMatrix.coeffRef(m, e) = orientation * gaussIntegralContributioni;
            std::tie(e,orientation) = getEdgeIndexOrientation(m,j);
            curlSolverMatrix.coeffRef(m, e) = orientation * gaussIntegralContributionj;
         }
      }

      // Add curlJ constraints for every element until the solver is happy.
      for(uint el=0; el<ionosphereGrid.elements.size() + ELEMENT_CONSTRAINT_REDUCTION; el++) {
         SphericalTriGrid::Element& element = ionosphereGrid.elements[el];
         Real A = ionosphereGrid.elementArea(el);

         int i=element.corners[0];
         int j=element.corners[1];
         int k=element.corners[2];

         Eigen::Vector3d r0(nodes[i].x.data());
         Eigen::Vector3d r1(nodes[j].x.data());
         Eigen::Vector3d r2(nodes[k].x.data());

         Eigen::Vector3d barycentre = getElementBarycentre(ionosphereGrid, el);

         // Make sure sign is correct (as edges are oriented)
         Real clockwise = r0.dot((r1-r0).cross(r2-r1));
         if(clockwise > 0) {
            clockwise = -1;
         } else {
            clockwise = 1;
         }

         auto [e,orientation] = getEdgeIndexOrientation(i,j);
         Real l1 = (r0 - barycentre).dot((r1-r0).normalized());
         Real l2 = (barycentre - r1).dot((r1-r0).normalized());
         curlSolverMatrix.insert(ionosphereGrid.nodes.size() + NODE_CONSTRAINT_REDUCTION + el, e) = orientation * (l1+l2);

         std::tie(e,orientation) = getEdgeIndexOrientation(j,k);
         l1 = (r1 - barycentre).dot((r2-r1).normalized());
         l2 = (barycentre - r2).dot((r2-r1).normalized());
         curlSolverMatrix.insert(ionosphereGrid.nodes.size() + NODE_CONSTRAINT_REDUCTION + el, e) = orientation * (l1+l2);

         std::tie(e,orientation) = getEdgeIndexOrientation(k,i);
         l1 = (r2 - barycentre).dot((r0-r2).normalized());
         l2 = (barycentre - r0).dot((r0-r2).normalized());
         curlSolverMatrix.insert(ionosphereGrid.nodes.size() + NODE_CONSTRAINT_REDUCTION + el, e) = orientation * (l1+l2);

         // Distribute FACs by area ratios
         Real A1 = getDualPolygonArea(ionosphereGrid, i);
         Real A2 = getDualPolygonArea(ionosphereGrid, j);
         Real A3 = getDualPolygonArea(ionosphereGrid, k);

         // NOTE: this is *not* yet the final right-hand side for the
         // divergence-free part here yet, as its values depend on the
         // solution of the curl-free part. Correction happens further
         // down.
         vRHS1[ionosphereGrid.nodes.size() + NODE_CONSTRAINT_REDUCTION + el] = clockwise * (nodes[element.corners[0]].parameters[ionosphereParameters::SOURCE] * 1./A1
              + nodes[element.corners[1]].parameters[ionosphereParameters::SOURCE] * 1./A2
              + nodes[element.corners[2]].parameters[ionosphereParameters::SOURCE] * 1./A3);
         vRHS2[ionosphereGrid.nodes.size() + NODE_CONSTRAINT_REDUCTION + el] = 0;

      }

      curlSolverMatrix.makeCompressed();

      if(writeSolverMtarix) {
         ofstream matrixOut("JSolverMatrix.txt");
         for(uint n=0; n<nodes.size() + NODE_CONSTRAINT_REDUCTION + ionosphereGrid.elements.size() + ELEMENT_CONSTRAINT_REDUCTION; n++) {
            for(uint m=0; m<edgeJCurl.size(); m++) {

               Real val=0;
               val = curlSolverMatrix.coeffRef(n, m);

               matrixOut << val << "\t";
            }
            matrixOut << endl;
         }
         if(!quiet) {
            cout << "--- CURL SOLVER MATRIX WRITTEN TO JSolverMatrix.txt ---" << endl;
         }
      }

      // Verify Euler characteristic of the mesh
      int Chi = nodes.size() - edgeJCurl.size() + ionosphereGrid.elements.size();
      cout << "Mesh has an euler characteristic of " << Chi << endl;

      // Solve curl-free currents.
      cout << "Solving divJ system" << endl;
#if NODE_CONSTRAINT_REDUCTION+ELEMENT_CONSTRAINT_REDUCTION != -2
      Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<Real>> solver;
#else
      Eigen::BiCGSTAB<Eigen::SparseMatrix<Real>> solver;
#endif
      solver.compute(curlSolverMatrix);
      vJ = solver.solve(vRHS2);
      cout << "... done with " << solver.iterations() << " iterations and remaining error " << solver.error() << "\n";

      for(uint e=0; e<edgeJDiv.size(); e++) {
         edgeJDiv[e] = vJ[e];
      }

      // Interpolate edge-localized J_CF to elements
      for(uint el=0; el<ionosphereGrid.elements.size(); el++) {
         // Average J from edge values, using Whitney 1-forms (DOI: 10.1145/1141911.1141991)
         Eigen::Vector3d j_cf = whitneyInterpolate(ionosphereGrid, el, edgeJDiv);

         elementCurlFreeCurrent[el] = j_cf;

         Eigen::Vector3d barycentre = getElementBarycentre(ionosphereGrid, el);
         Real MLT = atan2(barycentre[1], barycentre[0]) * 12 / M_PI + 12;

         // Note: The coefficients want to be looked up in A/km, so we multiply by 1000
         Real correction = pow(c4H(MLT)/c4P(MLT) * 1000*j_cf.norm(),1./(1.+c5P(MLT)-c5H(MLT))) / (1000*j_cf.norm());
         elementCorrectionFactors[el] = correction;
         //vRHS1[ionosphereGrid.nodes.size() + el] *= correction;
      }

      cout << "Solving curlJ system with " << nodes.size() << " nodes, " << ionosphereGrid.elements.size() << " elements and " << edgeJCurl.size() << " edges.\n";
      vJ = solver.solve(vRHS1);
      cout << "... done with " << solver.iterations() << " iterations and remaining error " << solver.error() << "\n";

      for(uint e=0; e<edgeJCurl.size(); e++) {
         edgeJCurl[e] = vJ[e];
      }

      // Now, likewise interpolate edge-localized J_DF to elements
      for(uint el=0; el<ionosphereGrid.elements.size(); el++) {
         // Effective curl-free current in this element
         Eigen::Vector3d j_df = whitneyInterpolate(ionosphereGrid, el, edgeJCurl);

         elementDivFreeCurrent[el] = j_df;
      }

      // Next, evaluate Sigma as a function of inplane-J and MLT
      #pragma omp parallel for
      for(uint n=0; n < nodes.size(); n++) {
         Eigen::Vector3d J(0,0,0);
         Eigen::Vector3d x(nodes[n].x.data());

         Real totalA=0;
         // Sum all incoming edges
         for(uint32_t el=0; el< nodes[n].numTouchingElements; el++) {
            Real A = ionosphereGrid.elementArea(nodes[n].touchingElements[el]);
            totalA += A;
            J += elementDivFreeCurrent[nodes[n].touchingElements[el]] * A;
         }
         J/=totalA;

         Real MLT = atan2(x[1], x[0]) * 12 / M_PI + 12;

         // Formula 33 from Juusola et al 2025
         // (in A/km)
         J *= 1000 * 5;
         Real SigmaH = c4H(MLT) * pow(J.norm(), c5H(MLT));
         Real SigmaP = c4P(MLT) * pow(J.norm(), c5P(MLT));

         nodes[n].parameters[ionosphereParameters::SIGMAP] = SigmaP;
         nodes[n].parameters[ionosphereParameters::SIGMAH] = SigmaH;
      }

      #pragma omp parallel for
      for(uint n=0; n<nodes.size(); n++) {

         if(nodes[n].openFieldLine > 0.5) {
            // Ignore polar cap conductivity.
            nodes[n].parameters[ionosphereParameters::SIGMAP] = 0;
            nodes[n].parameters[ionosphereParameters::SIGMAH] = 0;
         }
         // Also add solar contribution
         // Solar incidence parameter for calculating UV ionisation on the dayside
         Real coschi = nodes[n].x[0] / Ionosphere::innerRadius;
         Real chi = acos(coschi);
         Real qprime = altcos(chi);

         const Real F10_7 = 100;
         Real sigmaP_dayside = c1p * pow(F10_7, c2p) * pow(qprime, c3p);
         Real sigmaH_dayside = c1h * pow(F10_7, c2h) * pow(qprime, c3h);

         Real SigmaP = nodes[n].parameters[ionosphereParameters::SIGMAP];
         Real SigmaH = nodes[n].parameters[ionosphereParameters::SIGMAH];

         nodes[n].parameters[ionosphereParameters::SIGMAP] = sqrt(SigmaP*SigmaP + sigmaP_dayside*sigmaP_dayside +0.625*0.625);
         nodes[n].parameters[ionosphereParameters::SIGMAH] = sqrt(SigmaH*SigmaH + sigmaH_dayside*sigmaH_dayside +0.894*0.894);

         // TODO: We could instead directly calculate element conductivities using Whitney forms
         // and don't need to go via sigma averaging here.
         static const char epsilon[3][3][3] = {
            {{0,0,0},{0,0,1},{0,-1,0}},
            {{0,0,-1},{0,0,0},{1,0,0}},
            {{0,1,0},{-1,0,0},{0,0,0}}
         };

         Eigen::Vector3d b(nodes[n].x.data());
         b.normalize();
         if(nodes[n].x[2] >= 0) {
            b *= -1;
         }
         for(int i=0; i<3; i++) {
            for(int j=0; j<3; j++) {
               nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] = SigmaP * (((i==j)? 1. : 0.) - b[i]*b[j]);
               for(int k=0; k<3; k++) {
                  nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] -= SigmaH * epsilon[i][j][k]*b[k];
               }
            }
         }
      }

   } else {
      cerr << "Conductivity tensor " << sigmaString << " not implemented!" << endl;
      return 1;
   }

   // Write solver dependency matrix.
   if(writeSolverMtarix) {
      ofstream matrixOut("solverMatrix.txt");
      for(uint n=0; n<nodes.size(); n++) {
         for(uint m=0; m<nodes.size(); m++) {

            Real val=0;
            for(unsigned int d=0; d<nodes[n].numDepNodes; d++) {
               if(nodes[n].dependingNodes[d] == m) {
                  if(doPrecondition) {
                     val=nodes[n].dependingCoeffs[d] / nodes[n].dependingCoeffs[0];
                  } else {
                     val=nodes[n].dependingCoeffs[d];
                  }
               }
            }

            matrixOut << val << "\t";
         }
         matrixOut << endl;
      }
      if(!quiet) {
         cout << "--- SOLVER DEPENDENCY MATRIX WRITTEN TO solverMatrix.txt ---" << endl;
      }
   }

   ionosphereGrid.initSolver(true);

   // Try to solve the system.
   ionosphereGrid.isCouplingInwards=true;
   Ionosphere::solverPreconditioning = doPrecondition;
   Ionosphere::solverMaxFailureCount = 3;
   ionosphereGrid.rank = 0;
   int iterations, nRestarts;
   Real residual = std::numeric_limits<Real>::max(), minPotentialN, minPotentialS, maxPotentialN, maxPotentialS;

   // Measure solver timing
   timeval tStart, tEnd;
   gettimeofday(&tStart, NULL);
   ionosphereGrid.solve(iterations, nRestarts, residual, minPotentialN, maxPotentialN, minPotentialS, maxPotentialS);
   gettimeofday(&tEnd, NULL);
   double solverTime = (tEnd.tv_sec - tStart.tv_sec) + (tEnd.tv_usec - tStart.tv_usec) / 1000000.0;
   cout << "Own solver took " << solverTime << " seconds.\n";

   // Do the same solution using Eigen solver
   Eigen::SparseMatrix<Real> potentialSolverMatrix(nodes.size(), nodes.size());
   Eigen::VectorXd vRightHand(nodes.size()), vPhi(nodes.size());
   for(uint n=0; n<nodes.size(); n++) {

      for(uint m=0; m<nodes[n].numDepNodes; m++) {
         potentialSolverMatrix.insert(n, nodes[n].dependingNodes[m]) = nodes[n].dependingCoeffs[m];
      }

      vRightHand[n] = nodes[n].parameters[ionosphereParameters::SOURCE];
   }
   gettimeofday(&tStart, NULL);
   potentialSolverMatrix.makeCompressed();
   Eigen::BiCGSTAB<Eigen::SparseMatrix<Real> > solver;
   solver.compute(potentialSolverMatrix);
   vPhi = solver.solve(vRightHand);
   gettimeofday(&tEnd, NULL);
   cout << "... done with " << solver.iterations() << " iterations and remaining error " << solver.error() << "\n";
   solverTime = (tEnd.tv_sec - tStart.tv_sec) + (tEnd.tv_usec - tStart.tv_usec) / 1000000.0;
   cout << "Eigen solver took " << solverTime << " seconds.\n";

   if(!quiet) {
      cout << "Ionosphere solver: iterations " << iterations << " restarts " << nRestarts
         << " residual " << std::scientific << residual << std::defaultfloat
         << " potential min N = " << minPotentialN << " S = " << minPotentialS
         << " max N = " << maxPotentialN << " S = " << maxPotentialS
         << " difference N = " << maxPotentialN - minPotentialN << " S = " << maxPotentialS - minPotentialS
         << endl;
   } else {
      if(multipoleL == 0) {
         cout << std::scientific << residual << std::defaultfloat << std::endl;
      } else {
         // Actually corellate with our input multipole
         Real correlate=0;
         Real selfNorm=0;
         Real sphNorm =0;
         Real totalArea = 0;
         for(uint n=0; n<nodes.size(); n++) {
            double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
            double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

            Real area = 0;
            for(uint e=0; e<ionosphereGrid.nodes[n].numTouchingElements; e++) {
               area += ionosphereGrid.elementArea(ionosphereGrid.nodes[n].touchingElements[e]);
            }
            area /= 3.; // As every element has 3 corners, don't double-count areas

            totalArea += area;
            selfNorm += pow(nodes[n].parameters[ionosphereParameters::SOLUTION],2.) * area;
            sphNorm += pow(sph_legendre(multipoleL,fabs(multipolem),theta) * cos(multipolem*phi), 2.) * area;
            correlate += nodes[n].parameters[ionosphereParameters::SOLUTION] * sph_legendre(multipoleL,fabs(multipolem),theta) * cos(multipolem*phi) * area;
         }

         selfNorm = sqrt(selfNorm/totalArea);
         sphNorm = sqrt(sphNorm/totalArea);
         correlate /= totalArea * selfNorm * sphNorm;

         cout << std::scientific << correlate << std::defaultfloat << std::endl;
      }
   }

   // Write output
   vlsv::Writer outputFile;
   outputFile.open(outputFilename,MPI_COMM_WORLD,masterProcessID);
   ionosphereGrid.communicator = MPI_COMM_WORLD;
   ionosphereGrid.writingRank = 0;
   P::systemWriteName = std::vector<std::string>({"potato potato"});
   writeIonosphereGridMetadata(outputFile);

   // Data reducers
   DataReducer outputDROs;
   outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_fac", [](SBC::SphericalTriGrid& grid) -> std::vector<Real> {
         std::vector<Real> retval(grid.nodes.size());

         for (uint i = 0; i < grid.nodes.size(); i++) {
            Real area = 0;
            for (uint e = 0; e < grid.nodes[i].numTouchingElements; e++) {
               area += grid.elementArea(grid.nodes[i].touchingElements[e]);
            }
            area /= 3.; // As every element has 3 corners, don't double-count areas
            retval[i] = grid.nodes[i].parameters[ionosphereParameters::SOURCE] / area;
         }

         return retval;
   }));
   outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_source", [](SBC::SphericalTriGrid& grid) -> std::vector<Real> {
         std::vector<Real> retval(grid.nodes.size());

         for (uint i = 0; i < grid.nodes.size(); i++) {
            retval[i] = grid.nodes[i].parameters[ionosphereParameters::SOURCE];
         }

         return retval;
   }));
   outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_openclosed", [](SBC::SphericalTriGrid& grid) -> std::vector<Real> {
         std::vector<Real> retval(grid.nodes.size());

         for (uint i = 0; i < grid.nodes.size(); i++) {
            retval[i] = grid.nodes[i].openFieldLine;
         }

         return retval;
   }));
   outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_potential", [](SBC::SphericalTriGrid& grid)->std::vector<Real> {

         std::vector<Real> retval(grid.nodes.size());

         for(uint i=0; i<grid.nodes.size(); i++) {
            retval[i] = grid.nodes[i].parameters[ionosphereParameters::SOLUTION];
         }

         return retval;
   }));
   outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_EigenPotential", [&vPhi](SBC::SphericalTriGrid& grid)->std::vector<Real> {

         std::vector<Real> retval(grid.nodes.size());

         for(uint i=0; i<grid.nodes.size(); i++) {
            retval[i] = vPhi[i];
         }

         return retval;
   }));
   outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_residual", [](SBC::SphericalTriGrid& grid)->std::vector<Real> {

         std::vector<Real> retval(grid.nodes.size());

         for(uint i=0; i<grid.nodes.size(); i++) {
            retval[i] = grid.nodes[i].parameters[ionosphereParameters::RESIDUAL];
         }

         return retval;
   }));
   outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_sigmah", [](SBC::SphericalTriGrid& grid)->std::vector<Real> {

         std::vector<Real> retval(grid.nodes.size());

         for(uint i=0; i<grid.nodes.size(); i++) {
            retval[i] = grid.nodes[i].parameters[ionosphereParameters::SIGMAH];
         }

         return retval;
   }));
   outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_sigmap", [](SBC::SphericalTriGrid& grid)->std::vector<Real> {

         std::vector<Real> retval(grid.nodes.size());

         for(uint i=0; i<grid.nodes.size(); i++) {
            retval[i] = grid.nodes[i].parameters[ionosphereParameters::SIGMAP];
         }

         return retval;
   }));
   if(runCurlJSolver) {
      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereElement("ig_jFromCurlJ", [&edgeJCurl, &edgeIndex, &edgeLength,&elementDivFreeCurrent](
                  SBC::SphericalTriGrid& grid)->std::vector<Real> {

         std::vector<Real> retval(grid.elements.size()*3);

         for(uint i=0; i<grid.elements.size(); i++) {
            Eigen::Vector3d j=elementDivFreeCurrent[i];
            for(int n=0; n<3; n++) {
               retval[3*i + n] = j[n];
            }
         }

         return retval;
      }));

      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_jFromCurlJNode", [&](SBC::SphericalTriGrid& grid)->std::vector<Real> {

            std::vector<Real> retval(3*grid.nodes.size());

            for(uint n=0; n<grid.nodes.size(); n++) {
               Eigen::Vector3d J(0,0,0);

               Real totalA=0;
               // Sum all incoming edges
               for(uint32_t el=0; el< nodes[n].numTouchingElements; el++) {
                  Real A = grid.elementArea(nodes[n].touchingElements[el]);
                  totalA += A;
                  J += elementDivFreeCurrent[nodes[n].touchingElements[el]] * A;
               }
               J/=totalA;


               retval[3*n] = J[0];
               retval[3*n+1] = J[1];
               retval[3*n+2] = J[2];
            }

            return retval;
      }));
      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_jFromDivJNode", [&](SBC::SphericalTriGrid& grid)->std::vector<Real> {

            std::vector<Real> retval(3*grid.nodes.size());
            for(uint n=0; n<grid.nodes.size(); n++) {
               Eigen::Vector3d J(0,0,0);
               int numEdges=0;
               for(uint32_t el=0; el< nodes[n].numTouchingElements; el++) {
                  SphericalTriGrid::Element& element = ionosphereGrid.elements[nodes[n].touchingElements[el]];

                  int i=0, j=0;
                  for(int c=0; c<3; c++) {
                     if(element.corners[c] == n) {
                        i = element.corners[(c+1)%3];
                        j = element.corners[(c+2)%3];
                        break;
                     }
                  }

                  Eigen::Vector3d rn(nodes[n].x.data());
                  Eigen::Vector3d ri(nodes[i].x.data());
                  Eigen::Vector3d rj(nodes[j].x.data());

                  auto [e,orientation] = getEdgeIndexOrientation(i,n);
                  J += 0.5 * edgeJDiv[e] * orientation * (rn-ri).normalized();

                  std::tie(e,orientation) = getEdgeIndexOrientation(j,n);
                  J += 0.5 * edgeJDiv[e] * orientation * (rn-rj).normalized();
                  numEdges++;
               }

               J /= numEdges;

               retval[3*n] = J[0];
               retval[3*n+1] = J[1];
               retval[3*n+2] = J[2];
            }
            return retval;
      }));
      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereElement("ig_jFromDivJ", [&](SBC::SphericalTriGrid& grid)->std::vector<Real> {

            std::vector<Real> retval(3*grid.elements.size());

            for(uint el=0; el<grid.elements.size(); el++) {
               Eigen::Vector3d J = elementCurlFreeCurrent[el];

               retval[3*el] = J[0];
               retval[3*el+1] = J[1];
               retval[3*el+2] = J[2];
            }

            return retval;
      }));
      //outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereElement("ig_correctionFactor", [&](SBC::SphericalTriGrid& grid)->std::vector<Real> {

      //      std::vector<Real> retval(ionosphereGrid.elements.size());

      //      for(uint el=0; el<ionosphereGrid.elements.size(); el++) {
      //         retval[el]= elementCorrectionFactors[el];
      //      }
      //      return retval;
      //}));
      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_dualPolygonArea", [&](SBC::SphericalTriGrid& grid)->std::vector<Real> {
            std::vector<Real> retval(grid.nodes.size());

            for(uint n=0; n<grid.nodes.size(); n++) {
               retval[n] = getDualPolygonArea(grid, n);
            }

            return retval;
      }));
      //outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereElement("ig_elementArea", [&](SBC::SphericalTriGrid& grid)->std::vector<Real> {

      //      std::vector<Real> retval(ionosphereGrid.elements.size());

      //      for(uint el=0; el<ionosphereGrid.elements.size(); el++) {
      //         retval[el] = ionosphereGrid.elementArea(el);
      //      }
      //      return retval;
      //}));
   }

   for(unsigned int i=0; i<outputDROs.size(); i++) {
      outputDROs.writeIonosphereGridData(ionosphereGrid, "ionosphere", i, outputFile);
   }

   outputFile.close();
   if(!quiet) {
      cout << "--- OUTPUT WRITTEN TO " << outputFilename << " ---" << endl;
   }

   //cout << "--- DONE. ---" << endl;
   return 0;
}
