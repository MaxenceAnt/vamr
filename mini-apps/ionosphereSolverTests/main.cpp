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
   ionosphereGrid.initializeSphericalFibonacci(numNodes);
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

   // Set FACs
   if(facString == "constant") {
      for(uint n=0; n<nodes.size(); n++) {
         nodes[n].parameters[ionosphereParameters::SOURCE] = 1;
      }
   } else if(facString == "dipole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = 0;
         for(uint e=0; e<ionosphereGrid.nodes[n].numTouchingElements; e++) {
            area += ionosphereGrid.elementArea(ionosphereGrid.nodes[n].touchingElements[e]);
         }
         area /= 3.; // As every element has 3 corners, don't double-count areas

         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(1,0,theta) * cos(0*phi) * area;
      }
   } else if(facString == "quadrupole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = 0;
         for(uint e=0; e<ionosphereGrid.nodes[n].numTouchingElements; e++) {
            area += ionosphereGrid.elementArea(ionosphereGrid.nodes[n].touchingElements[e]);
         }
         area /= 3.; // As every element has 3 corners, don't double-count areas

         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(2,1,theta) * cos(1*phi) * area;
      }
   } else if(facString == "octopole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = 0;
         for(uint e=0; e<ionosphereGrid.nodes[n].numTouchingElements; e++) {
            area += ionosphereGrid.elementArea(ionosphereGrid.nodes[n].touchingElements[e]);
         }
         area /= 3.; // As every element has 3 corners, don't double-count areas

         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(3,2,theta) * cos(2*phi) * area;
      }
   } else if(facString == "hexadecapole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = 0;
         for(uint e=0; e<ionosphereGrid.nodes[n].numTouchingElements; e++) {
            area += ionosphereGrid.elementArea(ionosphereGrid.nodes[n].touchingElements[e]);
         }
         area /= 3.; // As every element has 3 corners, don't double-count areas

         nodes[n].parameters[ionosphereParameters::SOURCE] = sph_legendre(4,3,theta) * cos(3*phi) * area;
      }
   } else if(facString == "multipole") {
      for(uint n=0; n<nodes.size(); n++) {
         double theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         double phi = atan2(nodes[n].x[0], nodes[n].x[1]); // Longitude

         Real area = 0;
         for(uint e=0; e<ionosphereGrid.nodes[n].numTouchingElements; e++) {
            area += ionosphereGrid.elementArea(ionosphereGrid.nodes[n].touchingElements[e]);
         }
         area /= 3.; // As every element has 3 corners, don't double-count areas

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

         Real area = 0;
         for(uint e=0; e<ionosphereGrid.nodes[n].numTouchingElements; e++) {
            area += ionosphereGrid.elementArea(ionosphereGrid.nodes[n].touchingElements[e]);
         }
         area /= 3.; // As every element has 3 corners, don't double-count areas

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
   } else if(sigmaString == "curlJ" || sigmaString == "curlJmodified") {
      runCurlJSolver = true;

      // First, solve divergence-free inplane current system
      // Fill edge arrays
      for(uint32_t el=0; el< ionosphereGrid.elements.size(); el++) {
         SphericalTriGrid::Element& element = ionosphereGrid.elements[el];

         int i=element.corners[0],j=element.corners[1],k=element.corners[2];
         Eigen::Vector3d r0(nodes[i].x.data());
         Eigen::Vector3d r1(nodes[j].x.data());
         Eigen::Vector3d r2(nodes[k].x.data());

         // Edge length
         auto [e,orientation] = getEdgeIndexOrientation(i,j);
         edgeLength[e] = (r0-r1).norm() / 1e6;

         std::tie(e, orientation) = getEdgeIndexOrientation(j,k);
         edgeLength[e] = (r1-r2).norm() / 1e6;

         std::tie(e, orientation) = getEdgeIndexOrientation(k,i);
         edgeLength[e] = (r0-r2).norm() / 1e6;
      }

      // Eigen vector and matrix for solving
      Eigen::VectorXd vJ(edgeJCurl.size());
      Eigen::VectorXd vRHS1(nodes.size() + ionosphereGrid.elements.size() + 1); // Right hand side for divergence-free system
      Eigen::VectorXd vRHS2(nodes.size() + ionosphereGrid.elements.size() + 1); // Right hand side for curl-free system
      Eigen::SparseMatrix<Real> curlSolverMatrix(nodes.size() + ionosphereGrid.elements.size() + 1, edgeJCurl.size());

      // Divergence constraints
      for(uint m=0; m<nodes.size(); m++) {

         // Divergence
         vRHS1[m] = 0;
         vRHS2[m] = ionosphereGrid.nodes[m].parameters[ionosphereParameters::SOURCE];
         for(uint32_t el=0; el< nodes[m].numTouchingElements; el++) {
            SphericalTriGrid::Element& element = ionosphereGrid.elements[nodes[m].touchingElements[el]];

            // Find the two other nodes on this element
            int i=0,j=0;
            for(int c=0; c< 3; c++) {
               if(element.corners[c] == m) {
                  i=element.corners[(c+1)%3];
                  j=element.corners[(c+2)%3];
                  break;
               }
            }

            auto [e,orientation] = getEdgeIndexOrientation(m,i);
            curlSolverMatrix.coeffRef(m, e) = orientation;
            std::tie(e,orientation) = getEdgeIndexOrientation(m,j);
            curlSolverMatrix.coeffRef(m, e) = orientation;
         }
      }

      // Add curlJ constraints for every element until the solver is happy.
      for(uint el=0; el<ionosphereGrid.elements.size(); el++) {
         SphericalTriGrid::Element& element = ionosphereGrid.elements[el];
         Real A = ionosphereGrid.elementArea(el);

         int i=element.corners[0];
         int j=element.corners[1];
         int k=element.corners[2];

         Eigen::Vector3d r0(nodes[i].x.data());
         Eigen::Vector3d r1(nodes[j].x.data());
         Eigen::Vector3d r2(nodes[k].x.data());

         // Make sure sign is correct (as edges are oriented)
         Real clockwise = r0.dot((r1-r0).cross(r2-r1));
         if(clockwise > 0) {
            clockwise = 1;
         } else {
            clockwise = -1;
         }

         auto [e,orientation] = getEdgeIndexOrientation(i,j);
         curlSolverMatrix.insert(ionosphereGrid.nodes.size() + el, e) = orientation * edgeLength[e] / 100e3;

         std::tie(e,orientation) = getEdgeIndexOrientation(j,k);
         curlSolverMatrix.insert(ionosphereGrid.nodes.size() + el, e) = orientation * edgeLength[e] / 100e3;

         std::tie(e,orientation) = getEdgeIndexOrientation(k,i);
         curlSolverMatrix.insert(ionosphereGrid.nodes.size() + el, e) = orientation * edgeLength[e] / 100e3;

         // Distribute FACs by area ratios
         Real A1 = 0, A2 = 0, A3 = 0;
         for(uint e=0; e<ionosphereGrid.nodes[i].numTouchingElements; e++) {
            A1 += ionosphereGrid.elementArea(ionosphereGrid.nodes[i].touchingElements[e]);
         }
         for(uint e=0; e<ionosphereGrid.nodes[j].numTouchingElements; e++) {
            A2 += ionosphereGrid.elementArea(ionosphereGrid.nodes[j].touchingElements[e]);
         }
         for(uint e=0; e<ionosphereGrid.nodes[k].numTouchingElements; e++) {
            A3 += ionosphereGrid.elementArea(ionosphereGrid.nodes[k].touchingElements[e]);
         }
         vRHS1[ionosphereGrid.nodes.size() + el] = clockwise * (nodes[element.corners[0]].parameters[ionosphereParameters::SOURCE] * A/A1
               + nodes[element.corners[1]].parameters[ionosphereParameters::SOURCE] * A/A2
               + nodes[element.corners[2]].parameters[ionosphereParameters::SOURCE] * A/A3);
         vRHS2[ionosphereGrid.nodes.size() + el] = 0;

      }

      // Add Harmonic constraint (by pinning a cross-equator current to zero)
      for(const auto& [hash, edgeIdx] : edgeIndex) {
         uint32_t node1 = hash & 0xffffffff;
         uint32_t node2 = hash >> 32;

         // Find first edge that crosses the equator
         if(nodes[node1].x[0] * nodes[node2].x[0] < 0) {
            curlSolverMatrix.insert(curlSolverMatrix.rows()-1, edgeIdx) = 1.;
            vRHS1[vRHS1.size()-1] = 0.;
            vRHS2[vRHS2.size()-1] = 0.;
            break;
         }
      }

      curlSolverMatrix.makeCompressed();

      // Verify Euler characteristic of the mesh
      int Chi = nodes.size() - edgeJCurl.size() + ionosphereGrid.elements.size();
      cout << "Mesh has an euler characteristic of " << Chi << endl;

      //squareSolverMatrix = curlSolverMatrix.adjoint() * curlSolverMatrix;
      //vRightHand = curlSolverMatrix.adjoint() * vFAC;
      //squareSolverMatrix.makeCompressed();

      cout << "Solving curlJ system with " << nodes.size() << " nodes, " << ionosphereGrid.elements.size() << " elements and " << edgeJCurl.size() << " edges.\n";
      //Eigen::BiCGSTAB<Eigen::SparseMatrix<Real> > solver;
      Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<Real>> solver;
      //solver.compute(squareSolverMatrix);
      //vJ = solver.solve(vRightHand);
      solver.compute(curlSolverMatrix);
      vJ = solver.solve(vRHS1);
      cout << "... done with " << solver.iterations() << " iterations and remaining error " << solver.error() << "\n";

      for(uint e=0; e<edgeJCurl.size(); e++) {
         edgeJCurl[e] = vJ[e];
      }

      cout << "Solving divJ system with the same parameters" << endl;
      vJ = solver.solve(vRHS2);
      cout << "... done with " << solver.iterations() << " iterations and remaining error " << solver.error() << "\n";

      for(uint e=0; e<edgeJDiv.size(); e++) {
         edgeJDiv[e] = vJ[e];
      }

      bool modifiedModel = (sigmaString == "curlJmodified");

      // Next, evaluate Sigma as a function of inplane-J and MLT
      #pragma omp parallel for
      for(uint n=0; n < nodes.size(); n++) {
         Eigen::Vector3d J(0,0,0);
         Real A = 0;
         int numEdges = 0;
         Eigen::Vector3d x(nodes[n].x.data());

         // Sum all incoming edges
         for(uint32_t el=0; el< nodes[n].numTouchingElements; el++) {
            SphericalTriGrid::Element& element = ionosphereGrid.elements[nodes[n].touchingElements[el]];
            A += ionosphereGrid.elementArea(el);

            // Find the two other nodes on this element
            int i=0,j=0;
            for(int c=0; c< 3; c++) {
               if(element.corners[c] == n) {
                  i=element.corners[(c+1)%3];
                  j=element.corners[(c+2)%3];
                  break;
               }
            }

            Eigen::Vector3d xi(nodes[i].x.data());
            Eigen::Vector3d xj(nodes[j].x.data());
            auto [e, orientation] = getEdgeIndexOrientation(i, n);
            J += orientation * edgeJCurl[e] * (x - xi).normalized();
            std::tie(e,orientation) = getEdgeIndexOrientation(j,n);
            J += orientation * edgeJCurl[e] * (x - xj).normalized();

            numEdges += 2; // Note we have now double-counted the edges. But that's fine, we just divide by 2.
         }
         J/=numEdges;
         J/=A;

         Real MLT = atan2(x[1], x[0]) * 12 / M_PI + 12;
         Real rotJ = nodes[n].parameters[ionosphereParameters::SOURCE];

         // Lookup tables for fitting coefficients (note: MLT is in hours)
         std::function<Real(Real, Real)> c4P, c4H;
         std::function<Real(Real, Real)> c5P, c5H;
         // Uncorrected model
         if(!modifiedModel) {
            c4P = [](Real MLT, Real jsign) {
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

            c5P = [](Real MLT, Real jsign) {
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

            c4H = [](Real MLT, Real jsign) {
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

            c5H = [](Real MLT, Real jsign) {
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

            //c4P = [](Real MLT, Real jsign) {
            //   const Real values[] = {
            //      1.344, // 0
            //      1.161, // 3
            //      0.808, // 6
            //      0.640, // 9
            //      1.308, // 12
            //      0.439, // 15
            //      0.294, // 18
            //      0.815, // 21
            //   };

            //   int sector = MLT * 8. / 24.;
            //   Real interpolant = MLT * 8. / 24. - sector;
            //   return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%8];
            //};
            //c5P = [](Real MLT, Real jsign) {
            //   const Real values[] = {
            //      0.318, // 0
            //      0.332, // 3
            //      0.417, // 6
            //      0.384, // 9
            //      0.118, // 12
            //      0.401, // 15
            //      0.542, // 18
            //      0.423, // 21
            //   };

            //   int sector = MLT * 8. / 24.;
            //   Real interpolant = MLT * 8. / 24. - sector;
            //   return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%8];
            //};
            //c4H = [](Real MLT, Real jsign) {
            //   const Real values[] = {
            //      1.150, // 0
            //      1.347, // 3
            //      2.150, // 6
            //      2.153, // 9
            //      1.627, // 12
            //      0.637, // 15
            //      0.238, // 18
            //      0.670, // 21
            //   };

            //   int sector = MLT * 8. / 24.;
            //   Real interpolant = MLT * 8. / 24. - sector;
            //   return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%8];
            //};
            //c5H = [](Real MLT, Real jsign) {
            //   const Real values[] = {
            //      0.473, // 0
            //      0.441, // 3
            //      0.402, // 6
            //      0.358, // 9
            //      0.319, // 12
            //      0.361, // 15
            //      0.627, // 18
            //      0.573, // 21
            //   };

            //   int sector = MLT * 8. / 24.;
            //   Real interpolant = MLT * 8. / 24. - sector;
            //   return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%8];
            //};
         } else {
            // Modified model
            c4P = [](Real MLT, Real jsign) {
               const Real valuesPlus[] = {
                  1.344, 1.161, 0.808, 0.640, 1.308, 0.439, 0.294, 0.815
               };
               const Real valuesMinus[] = {
                  0.753, 0.690, 0.627, 0.564, 0.501, 0.439, 0.294, 0.815
               };
               int sector = MLT * 8. / 24.;
               Real interpolant = MLT * 8. / 24. - sector;
               if(jsign > 0) {
                  return (1.-interpolant)*valuesPlus[sector] + interpolant * valuesPlus[(sector+1)%8];
               } else {
                  return (1.-interpolant)*valuesMinus[sector] + interpolant * valuesMinus[(sector+1)%8];
               }
            };
            c5P = [](Real MLT, Real jsign) {
               const Real valuesPlus[] = {
                  0.318, 0.332, 0.417, 0.384, 0.118, 0.401, 0.542, 0.423
               };
               const Real valuesMinus[] = {
                  0.420, 0.416, 0.412, 0.408, 0.405, 0.401, 0.542, 0.423
               };
               int sector = MLT * 8. / 24.;
               Real interpolant = MLT * 8. / 24. - sector;
               if(jsign > 0) {
                  return (1.-interpolant)*valuesPlus[sector] + interpolant * valuesPlus[(sector+1)%8];
               } else {
                  return (1.-interpolant)*valuesMinus[sector] + interpolant * valuesMinus[(sector+1)%8];
               }
            };
            c4H = [](Real MLT, Real jsign) {
               const Real valuesPlus[] = {
                  1.150, 1.347, 2.150, 2.153, 1.627, 0.637, 0.238, 0.670
               };
               const Real valuesMinus[] = {
                  0.665, 0.659, 0.654, 0.648, 0.643, 0.637, 0.238, 0.670
               };
               int sector = MLT * 8. / 24.;
               Real interpolant = MLT * 8. / 24. - sector;
               if(jsign > 0) {
                  return (1.-interpolant)*valuesPlus[sector] + interpolant * valuesPlus[(sector+1)%8];
               } else {
                  return (1.-interpolant)*valuesMinus[sector] + interpolant * valuesMinus[(sector+1)%8];
               }
            };
            c5H = [](Real MLT, Real jsign) {
               const Real valuesPlus[] = {
                  0.473, 0.441, 0.402, 0.358, 0.319, 0.361, 0.627, 0.573
               };
               const Real valuesMinus[] = {
                  0.538, 0.502, 0.467, 0.431, 0.396, 0.361, 0.627, 0.573
               };
               int sector = MLT * 8. / 24.;
               Real interpolant = MLT * 8. / 24. - sector;
               if(jsign > 0) {
                  return (1.-interpolant)*valuesPlus[sector] + interpolant * valuesPlus[(sector+1)%8];
               } else {
                  return (1.-interpolant)*valuesMinus[sector] + interpolant * valuesMinus[(sector+1)%8];
               }
            };
         }

         // Formula 33 from Juusola et al 2025
         // (in A/km)
         J *= 1000;
         Real SigmaH = c4H(MLT,rotJ) * pow(J.norm(), c5H(MLT,rotJ));
         Real SigmaP = c4P(MLT,rotJ) * pow(J.norm(), c5P(MLT,rotJ));

         nodes[n].parameters[ionosphereParameters::SIGMAP] = SigmaP;
         nodes[n].parameters[ionosphereParameters::SIGMAH] = SigmaH;
      }

      // Read open/closed boundary from input file, to adjust sigmas in the polar regions
      vlsv::ParallelReader inVlsv;
      inVlsv.open(inputFile,MPI_COMM_WORLD,masterProcessID);
      readIonosphereNodeVariable(inVlsv, "ig_openclosed", ionosphereGrid, ionosphereParameters::ZPARAM); // NOTE: Abusing ZPARAM here, since the solver won't need it

      // Perform distance transform on the mesh
      // Here we have, as temporary variables:
      // ZPARAM -> Openclosed 1/0
      // ZZPARAM -> index of closest node (so far)
      // OPENCLOSEDIST -> distance to boundary
      //std::cerr << "Distance transform!" << std::endl << "[";
      //for(int n=0; n<nodes.size(); n++) {
      //   if(nodes[n].parameters[ionosphereParameters::ZPARAM] < 1.5) {
      //      nodes[n].parameters[ionosphereParameters::ZZPARAM] = n;
      //      nodes[n].parameters[ionosphereParameters::OPENCLOSEDIST] = 0;
      //   } else {
      //      nodes[n].parameters[ionosphereParameters::ZZPARAM] = -1;
      //      nodes[n].parameters[ionosphereParameters::OPENCLOSEDIST] = 6371e3;
      //   }
      //}

      //bool done=false;
      //while(!done) {
      //   done = true;
      //   for(int n=0; n<nodes.size(); n++) {
      //      if(nodes[n].parameters[ionosphereParameters::ZPARAM] < 1.5) {
      //         continue; // Skip closed nodes
      //      }
      //      Eigen::Vector3d x(nodes[n].x.data());

      //      for(int m=0; m<nodes[n].numTouchingElements; m++) {
      //         SphericalTriGrid::Element& element = ionosphereGrid.elements[nodes[n].touchingElements[m]];
      //         for(int c=0; c<3; c++) {
      //            int i = element.corners[c];
      //            if(i == n) {
      //               continue;
      //            }

      //            if(nodes[i].parameters[ionosphereParameters::ZPARAM] < 1.5) {
      //               // Closed nodes can be probed directly
      //               Eigen::Vector3d ox(nodes[i].x.data());
      //               Real distance = (ox - x).norm();
      //               if(distance < nodes[n].parameters[ionosphereParameters::OPENCLOSEDIST]) {
      //                  nodes[n].parameters[ionosphereParameters::OPENCLOSEDIST] = distance;
      //                  nodes[n].parameters[ionosphereParameters::ZZPARAM] = i;
      //                  done = false;
      //               }
      //            } else {
      //               // Open nodes require inferred distance
      //               // TODO: This should actually be geodetic distance, but maybe we can afford not to care
      //               if(nodes[i].parameters[ionosphereParameters::ZZPARAM] == -1) {
      //                  // This node doesn't even have a distance yet, skipping.
      //                  //done = false;
      //                  continue;
      //               }

      //               Eigen::Vector3d ox(nodes[ nodes[i].parameters[ionosphereParameters::ZZPARAM] ].x.data());
      //               Real distance = (ox - x).norm();
      //               if(distance < nodes[n].parameters[ionosphereParameters::OPENCLOSEDIST]) {
      //                  nodes[n].parameters[ionosphereParameters::OPENCLOSEDIST] = distance;
      //                  nodes[n].parameters[ionosphereParameters::ZZPARAM] = nodes[i].parameters[ionosphereParameters::ZZPARAM];
      //                  done = false;
      //               }
      //            }
      //         }
      //      }
      //   }
      //}
      //std::cerr << "]\nDistance transform done!" << std::endl;

      #pragma omp parallel for
      for(int n=0; n<nodes.size(); n++) {

         //// Adjust sigmas based on distance value
         //if(nodes[n].parameters[ionosphereParameters::OPENCLOSEDIST] > 300e3) { // TODO: Hardcoded 300km here
         //   Real alpha = (nodes[n].parameters[ionosphereParameters::OPENCLOSEDIST] - 300e3) / 300e3;
         //   nodes[n].parameters[ionosphereParameters::SIGMAP] *= exp(-alpha);
         //   nodes[n].parameters[ionosphereParameters::SIGMAH] *= exp(-alpha);
         //}
         if(nodes[n].parameters[ionosphereParameters::ZPARAM] > 1.5) {
            // Ignore polar cap conductivity.
            nodes[n].parameters[ionosphereParameters::SIGMAP] = 0;
            nodes[n].parameters[ionosphereParameters::SIGMAH] = 0;
         }
         // Also add solar contribution
         // Solar incidence parameter for calculating UV ionisation on the dayside
         Real coschi = nodes[n].x[0] / Ionosphere::innerRadius;
         if(coschi < 0) {
            coschi = 0;
         }
         const Real c1p = 0.585;
         const Real c2p = 0.582;
         const Real c3p = 0.267;
         const Real c1h = 1.854;
         const Real c2h = 0.409;
         const Real c3h = 0.353;
         const Real F10_7 = 100;
         Real sigmaP_dayside = c1p * pow(F10_7, c2p) * pow(coschi*coschi, c3p);
         Real sigmaH_dayside = c1h * pow(F10_7, c2h) * pow(coschi*coschi, c3h);

         Real SigmaP = nodes[n].parameters[ionosphereParameters::SIGMAP];
         Real SigmaH = nodes[n].parameters[ionosphereParameters::SIGMAH];

         nodes[n].parameters[ionosphereParameters::SIGMAP] = sqrt(SigmaP*SigmaP + sigmaP_dayside*sigmaP_dayside);
         nodes[n].parameters[ionosphereParameters::SIGMAH] = sqrt(SigmaH*SigmaH + sigmaH_dayside*sigmaH_dayside);

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
      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereElement("ig_jFromCurlJ", [&edgeJCurl, &edgeIndex, &edgeLength](
                  SBC::SphericalTriGrid& grid)->std::vector<Real> {

         std::vector<Real> retval(grid.elements.size()*3);

         for(uint i=0; i<grid.elements.size(); i++) {
            // Average J from edge values, using Whitney 1-forms (DOI: 10.1145/1141911.1141991)
            std::array<uint32_t, 3>& corners = grid.elements[i].corners;
            Real A = grid.elementArea(i);

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

            Eigen::Vector3d j = sqrt(A) * (o1*edgeLength[e1]*edgeJCurl[e1] * w1(barycentre) + o2*edgeLength[e2]*edgeJCurl[e2] * w2(barycentre) + o3*edgeLength[e3]*edgeJCurl[e3] *w3(barycentre));
            for(int n=0; n<3; n++) {
               retval[3*i + n] = j[n];
            }
         }

         return retval;
      }));

      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_jFromCurlJNode", [](SBC::SphericalTriGrid& grid)->std::vector<Real> {

            std::vector<Real> retval(3*grid.nodes.size());

            for(uint n=0; n<grid.nodes.size(); n++) {
               Eigen::Vector3d J(0,0,0);
               Real A = 0;

               int numEdges = 0;
               Eigen::Vector3d x(nodes[n].x.data());

               // Sum all incoming edges
               for(uint32_t el=0; el< nodes[n].numTouchingElements; el++) {
                  SphericalTriGrid::Element& element = ionosphereGrid.elements[nodes[n].touchingElements[el]];
                  A += ionosphereGrid.elementArea(el);

                  // Find the two other nodes on this element
                  int i=0,j=0;
                  for(int c=0; c< 3; c++) {
                     if(element.corners[c] == n) {
                        i=element.corners[(c+1)%3];
                        j=element.corners[(c+2)%3];
                        break;
                     }
                  }

                  Eigen::Vector3d xi(nodes[i].x.data());
                  Eigen::Vector3d xj(nodes[j].x.data());
                  auto [e, orientation] = getEdgeIndexOrientation(i, n);
                  J += orientation * edgeJCurl[e] * (x - xi).normalized();
                  std::tie(e,orientation) = getEdgeIndexOrientation(j,n);
                  J += orientation * edgeJCurl[e] * (x - xj).normalized();

                  numEdges += 2; // Note we have now double-counted the edges. But that's fine, we just divide by 2.
               }
               J/=numEdges;
               J/=A;

               retval[3*n] = J[0];
               retval[3*n+1] = J[1];
               retval[3*n+2] = J[2];
            }

            return retval;
      }));
      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_jFromDivJNode", [](SBC::SphericalTriGrid& grid)->std::vector<Real> {

            std::vector<Real> retval(3*grid.nodes.size());

            for(uint n=0; n<grid.nodes.size(); n++) {
               Eigen::Vector3d J(0,0,0);
               Real A = 0;

               int numEdges = 0;
               Eigen::Vector3d x(nodes[n].x.data());

               // Sum all incoming edges
               for(uint32_t el=0; el< nodes[n].numTouchingElements; el++) {
                  SphericalTriGrid::Element& element = ionosphereGrid.elements[nodes[n].touchingElements[el]];
                  A += ionosphereGrid.elementArea(el);

                  // Find the two other nodes on this element
                  int i=0,j=0;
                  for(int c=0; c< 3; c++) {
                     if(element.corners[c] == n) {
                        i=element.corners[(c+1)%3];
                        j=element.corners[(c+2)%3];
                        break;
                     }
                  }

                  Eigen::Vector3d xi(nodes[i].x.data());
                  Eigen::Vector3d xj(nodes[j].x.data());
                  auto [e, orientation] = getEdgeIndexOrientation(i, n);
                  J += orientation * edgeJDiv[e] * (x - xi).normalized();
                  std::tie(e,orientation) = getEdgeIndexOrientation(j,n);
                  J += orientation * edgeJDiv[e] * (x - xj).normalized();

                  numEdges += 2; // Note we have now double-counted the edges. But that's fine, we just divide by 2.
               }
               J/=numEdges;
               J/=A;

               retval[3*n] = J[0];
               retval[3*n+1] = J[1];
               retval[3*n+2] = J[2];
            }

            return retval;
      }));
      outputDROs.addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_openDistance", [](SBC::SphericalTriGrid& grid)->std::vector<Real> {

            std::vector<Real> retval(grid.nodes.size());

            for(uint i=0; i<grid.nodes.size(); i++) {
               retval[i] = nodes[i].parameters[ionosphereParameters::OPENCLOSEDIST];
            }

            return retval;
      }));
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
