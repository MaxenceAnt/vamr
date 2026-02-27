/*
 * This file is part of Vlasiator.
 * Copyright 2010-2016 Finnish Meteorological Institute
 *
 * For details of usage, see the COPYING file and read the "Rules of the Road"
 * at http://www.physics.helsinki.fi/vlasiator/
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/*!\file ionosphere.cpp
 * \brief Implementation of the class SysBoundaryCondition::Ionosphere to handle cells classified as sysboundarytype::IONOSPHERE.
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "ionosphere.h"
#include "../projects/project.h"
#include "../projects/projects_common.h"
#include "../vlasovsolver/vlasovmover.h"
#include "../fieldsolver/fs_common.h"
#include "../fieldsolver/ldz_magnetic_field.hpp"
#include "../fieldtracing/fieldtracing.h"
#include "../common.h"
#include "../object_wrapper.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include "../fieldtracing/fieldtracing.h"

#define Vec3d Eigen::Vector3d
#define cross_product(av,bv) (av).cross(bv)
#define dot_product(av,bv) (av).dot(bv)
#define vector_length(v) (v).norm()
#define normalize_vector(v) (v).normalized()

#ifdef DEBUG_VLASIATOR
#ifndef DEBUG_IONOSPHERE
#define DEBUG_IONOSPHERE
#endif
#endif
#ifdef DEBUG_SYSBOUNDARY
#ifndef DEBUG_IONOSPHERE
#define DEBUG_IONOSPHERE
#endif
#endif

namespace SBC {

   IonosphereBoundaryVDFmode boundaryVDFmode = FixedMoments;


   SphericalTriGrid ionosphereGrid; /*!< Ionosphere finite element grid */

   std::vector<IonosphereSpeciesParameters> Ionosphere::speciesParams;

   // Static ionosphere member variables
   Real Ionosphere::innerRadius;
   Real Ionosphere::radius;
   Real Ionosphere::recombAlpha; /*!< Recombination parameter, determining atmosphere ionizability (parameter) */
   Real Ionosphere::F10_7; /*!< Solar 10.7 Flux value (parameter) */
   Real Ionosphere::downmapRadius; /*!< Radius from which FACs are downmapped (RE) */
   Real Ionosphere::unmappedNodeRho; /*!< Electron density of ionosphere nodes that don't couple to the magnetosphere */
   Real Ionosphere::unmappedNodeTe; /*!< Electron temperature of ionosphere nodes that don't couple to the magnetosphere */
   Real Ionosphere::ridleyParallelConductivity; /*!< Constant parallel conductivity in the Ridley conductivity model */
   Real Ionosphere::couplingTimescale; /*!< Magnetosphere->Ionosphere coupling timescale (seconds) */
   Real Ionosphere::couplingInterval; /*!< Ionosphere update interval */
   int Ionosphere::solveCount; /*!< Counter of the number of solvings */
   Real Ionosphere::backgroundIonisation; /*!< Background ionisation due to stellar UV and cosmic rays */
   int  Ionosphere::solverMaxIterations;
   Real Ionosphere::solverRelativeL2ConvergenceThreshold;
   int Ionosphere::solverMaxFailureCount;
   Real Ionosphere::solverMaxErrorGrowthFactor;
   bool Ionosphere::solverPreconditioning;
   bool Ionosphere::solverUseMinimumResidualVariant;
   bool Ionosphere::solverToggleMinimumResidualVariant;
   Real Ionosphere::shieldingLatitude;
   enum Ionosphere::IonosphereConductivityModel Ionosphere::conductivityModel;

   // Offset field aligned currents so their sum is 0
   void SphericalTriGrid::offset_FAC() {

      if(nodes.size() == 0) {
         return;
      }

      // Separately make sure that both hemispheres are divergence-free
      Real northSum=0.;
      int northNum=0;
      Real southSum=0.;
      int southNum=0;

      for(uint n = 0; n<nodes.size(); n++) {
         if(nodes[n].x[2] > 0) {
            northSum += nodes[n].parameters[ionosphereParameters::SOURCE];
            northNum++;
         } else {
            southSum += nodes[n].parameters[ionosphereParameters::SOURCE];
            southNum++;
         }
      }

      northSum /= northNum;
      southSum /= southNum;

      for(uint n = 0; n<nodes.size(); n++) {
         if(nodes[n].x[2] > 0) {
            nodes[n].parameters[ionosphereParameters::SOURCE] -= northSum;
         } else {
            nodes[n].parameters[ionosphereParameters::SOURCE] -= southSum;
         }
      }
   }

   // Scale all nodes' coordinates so that they are situated on a spherical
   // shell with radius R
   void SphericalTriGrid::normalizeRadius(Node& n, Real R) {
      Real L = sqrt(n.x[0]*n.x[0] + n.x[1]*n.x[1] + n.x[2]*n.x[2]);
      for(int c=0; c<3; c++) {
         n.x[c] *= R/L;
      }
   }

   // Regenerate linking information between nodes and elements
   // Note: if this runs *before* stitchRefinementInterfaces(), there will be no
   // more information about t-junctions, so further stitiching won't work
   void SphericalTriGrid::updateConnectivity() {

      for(uint n=0; n<nodes.size(); n++) {
         nodes[n].numTouchingElements=0;

         for(uint e=0; e<elements.size(); e++) {
            for(int c=0; c<3; c++) {
               if(elements[e].corners[c] == n) {
                  nodes[n].touchingElements[nodes[n].numTouchingElements++]=e;
               }
            }
         }
      }
   }

   // Initialize base grid as a tetrahedron
   void SphericalTriGrid::initializeTetrahedron() {
      const static std::array<uint32_t, 3> seedElements[4] = {
         {1,2,3}, {1,3,4}, {1,4,2}, {2,4,3}
      };
      const static std::array<Real, 3> nodeCoords[4] = {
         {0,0,1.73205},
         {0,1.63299,-0.57735},
         {-1.41421,-0.816497,-0.57735},
         {1.41421,-0.816497,-0.57735}
      };

      // Create nodes
      // Additional nodes from table
      for(const auto& coords : nodeCoords) {
         Node newNode;
         newNode.x = coords;
         normalizeRadius(newNode, Ionosphere::innerRadius);
         nodes.push_back(newNode);
      }

      // Create elements
      for(const auto& seed : seedElements) {
         Element newElement;
         newElement.corners = seed;
         elements.push_back(newElement);
      }

      // Link elements to nodes
      updateConnectivity();
   }

   // Initialize base grid as an Octahedron
   void SphericalTriGrid::initializeOctahedron() {
      const static std::array<uint32_t, 3> seedElements[8] = {
         {0,1,2}, {0,2,3}, {0,3,4}, {0,4,1},
         {5,2,1}, {5,3,2}, {5,4,3}, {5,1,4},
      };
      const static std::array<Real, 3> nodeCoords[6] = {
         {0,0,1},
         {1,0,0},
         {0,1,0},
         {-1,0,0},
         {0,-10,0},
         {0,0,-1}
      };

      // Create nodes
      // Additional nodes from table
      for(const auto& coords : nodeCoords) {
         Node newNode;
         newNode.x = coords;
         normalizeRadius(newNode, Ionosphere::innerRadius);
         nodes.push_back(newNode);
      }

      // Create elements
      for(const auto& seed : seedElements) {
         Element newElement;
         newElement.corners = seed;
         elements.push_back(newElement);
      }

      // Link elements to nodes
      updateConnectivity();
   }

   // Initialize base grid as a icosahedron
   void SphericalTriGrid::initializeIcosahedron() {
      const static std::array<uint32_t, 3> seedElements[20] = {
        { 0, 2, 1}, { 0, 3, 2}, { 0, 4, 3}, { 0, 5, 4},
        { 0, 1, 5}, { 1, 2, 6}, { 2, 3, 7}, { 3, 4, 8},
        { 4, 5,9}, { 5, 1,10}, { 6, 2, 7}, { 7, 3, 8},
        { 8, 4,9}, {9, 5,10}, {10, 1, 6}, { 6, 7,11},
        { 7, 8,11}, { 8,9,11}, {9,10,11}, {10, 6,11}
      };
      const static std::array<Real, 3> nodeCoords[12] = {
        {        0,        0,  1.17557}, {  1.05146,        0, 0.525731},
        {  0.32492,      1.0, 0.525731}, {-0.850651, 0.618034, 0.525731},
        {-0.850651,-0.618034, 0.525731}, {  0.32492,     -1.0, 0.525731},
        { 0.850651, 0.618034,-0.525731}, { -0.32492,      1.0,-0.525731},
        { -1.05146,        0,-0.525731}, { -0.32492,     -1.0,-0.525731},
        { 0.850651,-0.618034,-0.525731}, {        0,        0, -1.17557}
      };

      // Create nodes
      // Additional nodes from table
      for(const auto& coords : nodeCoords) {
         Node newNode;
         newNode.x = coords;
         normalizeRadius(newNode, Ionosphere::innerRadius);
         nodes.push_back(newNode);
      }

      // Create elements
      for(const auto& seed : seedElements) {
         Element newElement;
         newElement.corners = seed;
         elements.push_back(newElement);
      }

      // Linke elements to nodes
      updateConnectivity();
   }

   // Spherical fibonacci base grid with arbitrary number of nodes n>8,
   // after Keinert et al 2015
   void SphericalTriGrid::initializeSphericalFibonacci(int n) {

      phiprof::Timer timer {"ionosphere-sphericalFibonacci"};
      // Golden ratio
      const Real Phi = (sqrt(5) +1.)/2.;

      auto madfrac = [](Real a, Real b) -> float {
         return a*b-floor(a*b);
      };

      // Forward spherical fibonacci mapping with n points
      auto SF = [madfrac,Phi](int i, int n) -> Vec3d {
         Real phi = 2*M_PI*madfrac(i, Phi-1.);
         Real z = 1. - (2.*i +1.)/n;
         Real sinTheta = sqrt(1 - z*z);
         return {cos(phi)*sinTheta, sin(phi)*sinTheta, z};
      };

      // Sample delaunay triangulation of the spherical fibonaccy grid around the given
      // point and return adjacent vertices
      auto SFDelaunayAdjacency = [SF,Phi](int j, int n) -> std::vector<int> {

         Real cosTheta = 1. - (2.*j+1.)/n;
         Real z = max(0., round(0.5*log(n * M_PI * sqrt(5) * (1.-cosTheta*cosTheta)) / log(Phi)));

         Vec3d nearestSample = SF(j,n);
         std::vector<int> nearestSamples;

         // Sample neighbourhood to find closest neighbours
         // Magic rainbow indexing
         for(int i=0; i<12; i++) {
            int r = i - floor(i/6)*6;
            int c = 5 - abs(5 - r*2) + floor((int)r/3);
            int k = j + (i < 6 ? +1 : -1) * (int)round(pow(Phi,z+c-2)/sqrt(5.));

            Vec3d currentSample = SF(k,n);
            Vec3d nearestToCurrentSample = currentSample - nearestSample;
            Real squaredDistance = dot_product(nearestToCurrentSample,nearestToCurrentSample);

            // Early reject by invalid index and distance
            if( k<0 || k>= n || squaredDistance > 5.*4.*M_PI / (sqrt(5) * n)) {
               continue;
            }

            nearestSamples.push_back(k);
         }

         // Make it delaunay
         std::vector<int> adjacentVertices;
         for(int i=0; i<(int)nearestSamples.size(); i++) {
            int k = nearestSamples[i];
            int kPrevious = nearestSamples[(i+nearestSamples.size()-1) % nearestSamples.size()];
            int kNext = nearestSamples[(i+1) % nearestSamples.size()];

            Vec3d currentSample = SF(k,n);
            Vec3d previousSample = SF(kPrevious, n);
            Vec3d nextSample = SF(kNext,n);

            if(dot_product(previousSample - nextSample, previousSample - nextSample) > dot_product(currentSample - nearestSample, currentSample-nearestSample)) {
               adjacentVertices.push_back(nearestSamples[i]);
            }
         }

         // Special case for the pole
         if( j == 0) {
            adjacentVertices.pop_back();
         }

         return adjacentVertices;
      };

      // Create nodes
      for(int i=0; i< n; i++) {
         Node newNode;

         Vec3d pos = SF(i,n);
         newNode.x = {pos[0], pos[1], pos[2]};
         normalizeRadius(newNode, Ionosphere::innerRadius);

         nodes.push_back(newNode);
      }

      // Create elements
      for(int i=0; i < n; i++) {
         std::vector<int> neighbours = SFDelaunayAdjacency(i,n);

         // Build a triangle fan around the neighbourhood
         for(uint j=0; j<neighbours.size(); j++) {
            if(neighbours[j] > i && neighbours[(j+1)%neighbours.size()] > i) { // Only triangles in "positive" direction to avoid double cover.
               Element newElement;
               newElement.corners = {(uint)i, (uint)neighbours[j], (uint)neighbours[(j+1)%neighbours.size()]};
               elements.push_back(newElement);
            }
         }
      }

      updateConnectivity();
   }

   void SphericalTriGrid::initializeGridFromFile(string pathString) {
      filesystem::path path = pathString;
      ifstream fi;
      fi.open(pathString.c_str());
      if (!fi.is_open()) {
         cerr << "(IONOSPHERE) Could not open file: " << pathString << endl;
         abort();
      }
      string line;
      if(path.extension() == ".obj"){
         while(getline(fi, line)){
            // Ignore all data other than vertices and faces
            if(!(line.rfind("v\t", 0) == 0 || line.rfind("v ", 0) == 0 || line.rfind("f", 0) == 0)){
               continue;
            }

            // Read vertices
            while(line.rfind("v ", 0) == 0){
               istringstream ss(line.substr(1));
               double num1, num2, num3;
               if (!(ss >> num1 >> num2 >> num3)) {
                  cerr << "(IONOSPHERE) Error reading vertex information of line \"" << line <<"\" in " << pathString << endl;
                  abort();
               }
               Node newNode;
               newNode.x = {num1, num2, num3};
               normalizeRadius(newNode, Ionosphere::innerRadius);
               nodes.push_back(newNode);
               getline(fi, line);
            }

            int length = nodes.size();
            // Read faces, support negative number specification
            while(line.rfind("f", 0) == 0){
               istringstream ss(line.substr(1));
               string faceArg;
               std::vector<int> vertexIndices;
               // Ignore normal and texture vertices
               while(ss >> faceArg){
                  istringstream fss(faceArg);
                  int v;
                  if (!(fss >> v)) {
                     cerr << "(IONOSPHERE) Error reading face information of line \"" << line <<"\" in " << pathString << endl;
                     abort();
                  }
                  // Support negative indices (indices are 1-indexed)
                  if(v < 0){ 
                     v = length + v;
                  } else {
                     v = v - 1;
                  }
                  if(v < 0 || v >= length) {
                     cerr << "(IONOSPHERE) Invalid vertex index (" << v << ") specified in \"" << line <<"\" in " << pathString << endl;
                     abort();
                  }
                  vertexIndices.push_back(v);
               }
               if(vertexIndices.size() != 3){
                  cerr << "(IONOSPHERE) Too many vertex indices (" << vertexIndices.size() << ") specified in \"" << line <<"\" in " << pathString << " (Only triangulated meshes are supported)" <<endl;
                  abort();
               }
               Element newElement;
               newElement.corners = std::array<uint32_t,3>{vertexIndices[0], vertexIndices[1], vertexIndices[2]};
               elements.push_back(newElement);
               getline(fi, line);
            }
         }
         
         if(nodes.size() == 0){
            cerr << "(IONOSPHERE) Error reading nodes in \"" << pathString << "\", expected a non-zero number of nodes to be specified." << endl;
            abort();
         }

         if(elements.size() == 0){
            cerr << "(IONOSPHERE) Error reading faces in \"" << pathString << "\", expected a non-zero number of faces to be specified." << endl;
            abort();
         }
      } else if (path.extension() == ".vtk"){
         if(!getline(fi, line)){
            cerr << "(IONOSPHERE) Error reading version string in " << pathString << endl;
            abort();
         }
         if(!(line.rfind("# vtk DataFile Version ", 0) == 0)) {
            cerr << "(IONOSPHERE) Expected mandatory VTK version string, obtained \"" << line << "\" in " << pathString << endl;
            abort();
         }
         float version = stof(line.substr(23));
         if(version > 4.2){
            cerr << "(IONOSPHERE) VTK version unsupported, expected legacy version less than 4.2, instead obtained " << version << " in " << pathString << endl;
            abort();
         }
         if(!getline(fi, line)){
            cerr << "(IONOSPHERE) Error reading mandatory description string in " << pathString << endl;
            abort();
         }
         if(!getline(fi,line)){
            cerr << "(IONOSPHERE) Error reading mandatory data type string in " << pathString << ", ASCII or BINARY data not specified." << endl;
            abort();
         }
         if(line != "ASCII"){
            cerr << "(IONOSPHERE) Only ASCII VTK data is supported, obtained " << line << endl;
            abort();
         }


         if(getline(fi, line)){
            stringstream ss(line);
            string dataset;
            string data;
            if(ss >> dataset >> data){
               if(dataset != "DATASET" || data != "UNSTRUCTURED_GRID"){
                  cerr << "(IONOSPHERE) Could not find DATASET specification in " << pathString << endl;
                  abort();
               }
            }
         } else {
            cerr << "(IONOSPHERE) Error reading mandatory DATASET string in " << pathString << endl;
            abort();
         }

         if(getline(fi,line)) {
            std::vector<Real> coords;
            stringstream pss(line);
            string points;
            int size;
            string type;
            
            if(!(pss >> points >> size >> type)){
               cerr << "(IONOSPHERE) Could not read POINTS field \"" << line << "\"" << " in " << pathString << endl;
               abort();
            }
            
            if(!(points == "POINTS")){
               cerr << "(IONOSPHERE) Mandatory POINTS field not found, obtained " << line << "\" in " << pathString << endl;
               abort();
            }

            if(type != "float" && type != "double"){
               cerr << "(IONOSPHERE) Only float or double are supported, obtained \"" << type << "\" in " << pathString << endl;
               abort();
            }

            while(getline(fi,line) && all_of(line.begin(), line.end(), [](char c){
               return c == 'e' || c == 'E' || c == '+' || c == '.' || c == ' ' || c == '-' || isdigit(c);
            })){
               stringstream css(line);
               double x;
               while(css >> x){
                  coords.push_back(x);
               }
            }

            if(coords.size() != size*3) {
               cerr << "(IONOSPHERE) Number of coordinates in POINTS field (" << size*3 << ") does not match number of coordinates found (" << coords.size() << ") in " << pathString << endl;
               abort();
            }


            for(int i = 0; i < coords.size(); i+=3){
               Node newNode;
               newNode.x = {coords[i], coords[i+1], coords[i+2]};
               normalizeRadius(newNode, Ionosphere::innerRadius);
               nodes.push_back(newNode);
            }

         } else {
            cerr << "(IONOSPHERE) Could not read POINTS field in " << pathString << endl;
               abort();
         }

         if(!fi.eof()){
            stringstream css(line);
            string cells;
            int cellNum;
            int size;

            if(!(css >> cells >> cellNum >> size)){
               cerr << "(IONOSPHERE) Could not read CELLS field \"" << line << "\"" << " in " << pathString << endl;
               abort();
            }

            if(!(cells == "CELLS")){
               cerr << "(IONOSPHERE) Mandatory CELLS field not found, obtained " << line << "\" in " << pathString << endl;
               abort();
            }

            if(!(cellNum*4 == size)){
               cerr << "(IONOSPHERE) Incorrect number of entries for the corresponding number of cells, obtained " << line << "\" in " << pathString << endl;
               abort();
            }

            while(getline(fi,line) && all_of(line.begin(), line.end(), [](char c){
               return c == ' ' || isdigit(c);
            })){
               stringstream css(line);
               int t, a, b, c;
               while(css >> t >> a >> b >> c){
                  if(!(t == 3)){
                     cerr << "(IONOSPHERE) Non-triangular cell encountered, \"" << line << "\" in " << pathString << endl;
                     abort();
                  }
                  for(int v : {a, b, c}){
                     if(v < 0 || v >= nodes.size()){
                        cerr << "(IONOSPHERE) Error vertex number out of bounds, " << v << " in \"" << line << "\" in " << pathString << endl;
                        abort();
                     }
                  }
                  Element newElement;
                  newElement.corners = {a, b, c};
                  elements.push_back(newElement);
               }
            }

            if(elements.size() != cellNum) {
               cerr << "(IONOSPHERE) Number of cells does not match file, expected " << cellNum << ", obtained " << elements.size() << " in " << pathString << endl;
               abort();
            }
         } else {
            cerr << "(IONOSPHERE) Could not read CELLS field \"" << line << "\"" << " in " << pathString << endl;
            abort();
         }

      } else {
         cerr << "(IONOSPHERE) Unknown ionosphere grid mesh file format " << path.extension() << endl;
         abort();
      }

      updateConnectivity();
   }

   // Find the neighbouring element of the one with index e, that is sharing the
   // two corner nodes n1 and n2
   //
   //            2 . . . . . . . .*
   //           /  \             .
   //          /    \   neigh   .
   //         /      \   bour  .
   //        /   e    \       .
   //       /          \     .
   //      /            \   .
   //     /              \ .
   //    0----------------1
   //
   int32_t SphericalTriGrid::findElementNeighbour(uint32_t e, int n1, int n2) {
      Element& el = elements[e];

      Node& node1 = nodes[el.corners[n1]];
      Node& node2 = nodes[el.corners[n2]];

      for(uint n1e=0; n1e<node1.numTouchingElements; n1e++) {
         if(node1.touchingElements[n1e] == e) continue; // Skip ourselves.

         for(uint n2e=0; n2e<node2.numTouchingElements; n2e++) {
            if(node1.touchingElements[n1e] == node2.touchingElements[n2e]) {
               return node1.touchingElements[n1e];
            }
         }
      }

      // No neighbour found => Apparently, the neighbour is refined and doesn't
      // exist at this scale. Good enough for us.
      return -1;
   }

   // Find the mesh node closest to the given coordinates.
   uint32_t SphericalTriGrid::findNodeAtCoordinates(std::array<Real,3> x) {

      // Project onto sphere
      Real L=sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
      for(int c=0; c<3; c++) {
         x[c] *= Ionosphere::innerRadius/L;
      }

      uint32_t node = 0;
      uint32_t nextNode = 0;

      // TODO: For spherical fibonacci meshes, this can be accelerated by
      // doing an iSF lookup

      // Iterate through nodes to find the closest one
      while(true) {

         node = nextNode;

         // This nodes' distance to our target point
         std::array<Real, 3> deltaX({x[0]-nodes[node].x[0],
               x[1]-nodes[node].x[1],
               x[2]-nodes[node].x[2]});
         Real minDist=sqrt(deltaX[0]*deltaX[0] + deltaX[1]*deltaX[1] + deltaX[2]*deltaX[2]);

         // Iterate through our neighbours
         for(uint i=0; i<nodes[node].numTouchingElements; i++) {
            for(int j=0; j<3; j++) {
               uint32_t thatNode = elements[nodes[node].touchingElements[i]].corners[j];
               if(thatNode == node || thatNode == nextNode) {
                  continue;
               }

               // If it is closer, continue there.
               deltaX = {x[0]-nodes[thatNode].x[0],
                  x[1]-nodes[thatNode].x[1],
                  x[2]-nodes[thatNode].x[2]};
               Real thatDist = sqrt(deltaX[0]*deltaX[0] + deltaX[1]*deltaX[1] + deltaX[2]*deltaX[2]);
               if(thatDist < minDist) {
                  minDist = thatDist;
                  nextNode = thatNode;
               }
            }
         }

         // Didn't find a closer one, use this one.
         if(nextNode == node) {
            break;
         }
      }

      return node;
   }

   // Subdivide mesh within element e
   // The element gets replaced by four new ones:
   //
   /*            2                      2
   //           /  \                   /  \
   //          /    \                 / 2  \
   //         /      \               /      \
   //        /        \     ==>     2--------1
   //       /          \           / \  3  /  \
   //      /            \         / 0 \   / 1  \
   //     /              \       /     \ /      \
   //    0----------------1     0-------0--------1
   */
   // And three new nodes get created at the interfaces,
   // unless they already exist.
   // The new center element (3) replaces the old parent element in place.
   void SphericalTriGrid::subdivideElement(uint32_t e) {

      phiprof::Timer timer {"ionosphere-subdivideElement"};
      Element& parentElement = elements[e];

      // 4 new elements
      std::array<Element,4> newElements;
      for(int i=0; i<4; i++) {
         newElements[i].refLevel = parentElement.refLevel + 1;
      }

      // (up to) 3 new nodes
      std::array<uint32_t,3> edgeNodes;
      for(int i=0; i<3; i++) { // Iterate over the edges of the triangle

         // Taking the two nodes on that edge
         Node& n1 = nodes[parentElement.corners[i]];
         Node& n2 = nodes[parentElement.corners[(i+1)%3]];

         // Find the neighbour in that direction
         int32_t ne = findElementNeighbour(e, i, (i+1)%3);

         if(ne == -1) { // Neighbour is refined already, node should already exist.

            // Find it.
            int32_t insertedNode = -1;

            // First assemble a list of candidates from all elements touching
            // that corner at the next refinement level
            std::set<uint32_t> candidates;
            for(uint en=0; en< n1.numTouchingElements; en++) {
               if(elements[n1.touchingElements[en]].refLevel == parentElement.refLevel + 1) {
                  for(int k=0; k<3; k++) {
                     candidates.emplace(elements[n1.touchingElements[en]].corners[k]);
                  }
               }
            }
            // Then match that list from the second corner
            for(uint en=0; en< n2.numTouchingElements; en++) {
               if(elements[n2.touchingElements[en]].refLevel == parentElement.refLevel + 1) {
                  for(int k=0; k<3; k++) {
                     if(candidates.count(elements[n2.touchingElements[en]].corners[k]) > 0) {
                        insertedNode = elements[n2.touchingElements[en]].corners[k];
                     }
                  }
               }
            }
            if(insertedNode == -1) {
               cerr << "(IONOSPHERE) Warning: did not find neighbouring split node when trying to refine "
                  << "element " << e << " on edge " << i << " with nodes (" << parentElement.corners[0]
                  << ", " << parentElement.corners[1] << ", " << parentElement.corners[2] << ")" << endl;
               insertedNode = 0;
            }

            // Double-check that this node currently has 4 touching elements
            if(nodes[insertedNode].numTouchingElements != 4) {
               cerr << "(IONOSPHERE) Warning: mesh topology screwup when refining: node "
                  << insertedNode << " is touching " << nodes[insertedNode].numTouchingElements
                  << " elements, should be 4." << endl;
            }

            // Add the other 2
            nodes[insertedNode].touchingElements[4] = elements.size() + i;
            nodes[insertedNode].touchingElements[5] = elements.size() + (i+1)%3;

            // Now that node touches 6 elements.
            nodes[insertedNode].numTouchingElements=6;

            edgeNodes[i] = insertedNode;

         } else {       // Neighbour is not refined, add a node here.
            Node newNode;

            // Node coordinates are in the middle of the two parents
            for(int c=0; c<3; c++) {
               newNode.x[c] = 0.5 * (n1.x[c] + n2.x[c]);
            }
            // Renormalize to sit on the circle
            normalizeRadius(newNode, Ionosphere::innerRadius);

            // This node has four touching elements: the old neighbour and 3 of the new ones
            newNode.numTouchingElements = 4;
            newNode.touchingElements[0] = ne;
            newNode.touchingElements[1] = e; // Center element
            newNode.touchingElements[2] = elements.size() + i;
            newNode.touchingElements[3] = elements.size() + (i+1)%3;

            nodes.push_back(newNode);
            edgeNodes[i] = nodes.size()-1;
         }
      }

      // Now set the corners of the new elements
      newElements[0].corners[0] = parentElement.corners[0];
      newElements[0].corners[1] = edgeNodes[0];
      newElements[0].corners[2] = edgeNodes[2];
      newElements[1].corners[0] = edgeNodes[0];
      newElements[1].corners[1] = parentElement.corners[1];
      newElements[1].corners[2] = edgeNodes[1];
      newElements[2].corners[0] = edgeNodes[2];
      newElements[2].corners[1] = edgeNodes[1];
      newElements[2].corners[2] = parentElement.corners[2];
      newElements[3].corners[0] = edgeNodes[0];
      newElements[3].corners[1] = edgeNodes[1];
      newElements[3].corners[2] = edgeNodes[2];

      // And references of the corners are replaced to point
      // to the new child elements
      for(int n=0; n<3; n++) {
         Node& cornerNode = nodes[parentElement.corners[n]];
         for(uint i=0; i< cornerNode.numTouchingElements; i++) {
            if(cornerNode.touchingElements[i] == e) {
               cornerNode.touchingElements[i] = elements.size() + n;
            }
         }
      }

      // The center element replaces the original one
      elements[e] = newElements[3];
      // Insert the other new elements at the end of the list
      for(int i=0; i<3; i++) {
         elements.push_back(newElements[i]);
      }
   }


   // Fractional energy dissipation rate for a isotropic beam, based on Rees (1963), figure 1
   static Real ReesIsotropicLambda(Real x) {
      static const Real P[7] = { -11.639, 32.1133, -30.8543, 14.6063, -6.3375, 0.6138, 1.4946};
      Real lambda = (((((P[0] * x + P[1])*x +P[2])*x+ P[3])*x + P[4])*x +P[5])* x+P[6];
      if(x > 1. || lambda < 0) {
         return 0;
      }
      return lambda;
   }


   // Energy dissipasion function based on Sergienko & Ivanov (1993), eq. A2
   static Real SergienkoIvanovLambda(Real E0, Real Chi) {

      struct SergienkoIvanovParameters {
         Real E; // in eV
         Real C1;
         Real C2;
         Real C3;
         Real C4;
      };

      const static SergienkoIvanovParameters SIparameters[] = {
         {50,  0.0409,   1.072, -0.0641, -1.054},
         {100, 0.0711,   0.899, -0.171,  -0.720},
         {500, 0.130,    0.674, -0.271,  -0.319},
         {1000,0.142,    0.657, -0.277,  -0.268}
      };

      Real C1=0;
      Real C2=0;
      Real C3=0;
      Real C4=0;
      if(E0 <= SIparameters[0].E) {
         C1 = SIparameters[0].C1;
         C2 = SIparameters[0].C2;
         C3 = SIparameters[0].C3;
         C4 = SIparameters[0].C4;
      } else if (E0 >= SIparameters[3].E) {
         C1 = SIparameters[3].C1;
         C2 = SIparameters[3].C2;
         C3 = SIparameters[3].C3;
         C4 = SIparameters[3].C4;
      } else {
         for(int i=0; i<3; i++) {
            if(SIparameters[i].E < E0 && SIparameters[i+1].E > E0) {
               Real interp = (E0 - SIparameters[i].E) / (SIparameters[i+1].E - SIparameters[i].E);
               C1 = (1.-interp) * SIparameters[i].C1 + interp * SIparameters[i+1].C1;
               C2 = (1.-interp) * SIparameters[i].C2 + interp * SIparameters[i+1].C2;
               C3 = (1.-interp) * SIparameters[i].C3 + interp * SIparameters[i+1].C3;
               C4 = (1.-interp) * SIparameters[i].C4 + interp * SIparameters[i+1].C4;
            }
         }
      }
      return (C2 + C1*Chi)*exp(C4*Chi + C3*Chi*Chi);
   }

   /* Read atmospheric model file in MSIS format.
    * Based on the table data, precalculate and fill the ionisation production lookup table
    */
   void SphericalTriGrid::readAtmosphericModelFile(const char* filename) {

      phiprof::Timer timer {"ionosphere-readAtmosphericModelFile"};
      // These are the only height values (in km) we are actually interested in
      static const float alt[numAtmosphereLevels] = {
         66, 68, 71, 74, 78, 82, 87, 92, 98, 104, 111,
         118, 126, 134, 143, 152, 162, 172, 183, 194
      };

      // Open file, read in
      ifstream in(filename);
      if(!in) {
         cerr << "(ionosphere) WARNING: Atmospheric Model file " << filename << " could not be opened: " <<
            strerror(errno) << endl
            << "(ionosphere) All atmospheric values will be zero, and there will be no ionization!" << endl;
      }
      int altindex = 0;
      Real integratedDensity = 0;
      Real prevDensity = 0;
      Real prevAltitude = 0;
      std::vector<std::array<Real, 5>> MSISvalues;
      while(in) {
         Real altitude, massdensity, Odensity, N2density, O2density, neutralTemperature;
         in >> altitude >> Odensity >> N2density >> O2density >> massdensity >> neutralTemperature;

         integratedDensity += (altitude - prevAltitude) *1000 * 0.5 * (massdensity + prevDensity);
         // Ion-neutral scattering frequencies (from Schunk and Nagy, 2009, Table 4.5)
         Real nui = 1e-17*(3.67*Odensity + 5.14*N2density + 2.59*O2density);
         // Elctron-neutral scattering frequencies (Same source, Table 4.6)
         Real nue = 1e-17*(8.9*Odensity + 2.33*N2density + 18.2*O2density);
         prevAltitude = altitude;
         prevDensity = massdensity;
         MSISvalues.push_back({altitude, massdensity, nui, nue, integratedDensity});
      }

      // Iterate through the read data and linearly interpolate
      for(unsigned int i=1; i<MSISvalues.size(); i++) {
         Real altitude = MSISvalues[i][0];

         // When we encounter one of our reference layers, record its values
         while(altindex < numAtmosphereLevels && altitude >= alt[altindex]) {
            Real interpolationFactor = (alt[altindex] - MSISvalues[i-1][0]) / (MSISvalues[i][0] - MSISvalues[i-1][0]);

            AtmosphericLayer newLayer;
            newLayer.altitude = alt[altindex]; // in km
            newLayer.density = fmax((1.-interpolationFactor) * MSISvalues[i-1][1] + interpolationFactor * MSISvalues[i][1], 0.); // kg/m^3
            newLayer.depth = fmax((1.-interpolationFactor) * MSISvalues[i-1][4] + interpolationFactor * MSISvalues[i][4], 0.); // kg/m^2

            newLayer.nui = fmax((1.-interpolationFactor) * MSISvalues[i-1][2] + interpolationFactor * MSISvalues[i][2], 0.); // m^-3 s^-1
            newLayer.nue = fmax((1.-interpolationFactor) * MSISvalues[i-1][3] + interpolationFactor * MSISvalues[i][3], 0.); // m^-3 s^-1
            atmosphere[altindex++] = newLayer;
         }
      }

      // Now we have integrated density from the bottom of the atmosphere in the depth field.
      // Flip it around.
      for(int h=0; h<numAtmosphereLevels; h++) {
         atmosphere[h].depth = integratedDensity - atmosphere[h].depth;
      }

      // Calculate Hall and Pedersen conductivity coefficient based on charge carrier density
      const Real Bval = 5e-5; // TODO: Hardcoded B strength here?
      const Real NO_gyroFreq = physicalconstants::CHARGE * Bval / (31*physicalconstants::MASS_PROTON); // Ion (NO+) gyration frequency
      const Real e_gyroFreq = physicalconstants::CHARGE * Bval / (physicalconstants::MASS_ELECTRON); // Elctron gyration frequency
      for(int h=0; h<numAtmosphereLevels; h++) {
         Real sigma_i = physicalconstants::CHARGE*physicalconstants::CHARGE / ((31. * physicalconstants::MASS_PROTON)  * atmosphere[h].nui);
         Real sigma_e = physicalconstants::CHARGE*physicalconstants::CHARGE / (physicalconstants::MASS_ELECTRON  * atmosphere[h].nue);
         atmosphere[h].pedersencoeff = sigma_i * (atmosphere[h].nui * atmosphere[h].nui)/(atmosphere[h].nui*atmosphere[h].nui + NO_gyroFreq*NO_gyroFreq)
            + sigma_e *(atmosphere[h].nue * atmosphere[h].nue)/(atmosphere[h].nue*atmosphere[h].nue + e_gyroFreq*e_gyroFreq);
         atmosphere[h].hallcoeff = -sigma_i * (atmosphere[h].nui * NO_gyroFreq)/(atmosphere[h].nui*atmosphere[h].nui + NO_gyroFreq*NO_gyroFreq)
            + sigma_e *(atmosphere[h].nue * e_gyroFreq)/(atmosphere[h].nue*atmosphere[h].nue + e_gyroFreq*e_gyroFreq);

         atmosphere[h].parallelcoeff = sigma_e;
      }


      // Energies of particles that sample the production array
      // are logspace-distributed from 10^-1 to 10^2.3 keV
      std::array< Real, SBC::productionNumParticleEnergies+1 > particle_energy; // In KeV
      for(int e=0; e<SBC::productionNumParticleEnergies; e++) {
         // TODO: Hardcoded constants. Make parameter?
         particle_energy[e] = pow(10.0, -1.+e*(2.3+1.)/(SBC::productionNumParticleEnergies-1));
      }
      particle_energy[SBC::productionNumParticleEnergies] = 2*particle_energy[SBC::productionNumParticleEnergies-1] - particle_energy[SBC::productionNumParticleEnergies-2];

      // Precalculate scattering rates
      const Real eps_ion_keV = 0.035; // Energy required to create one ion
      std::array< std::array< Real, numAtmosphereLevels >, SBC::productionNumParticleEnergies > scatteringRate;
      for(int e=0;e<SBC::productionNumParticleEnergies; e++) {

         Real electronRange=0.;
         Real rho_R=0.;
         switch(ionizationModel) {
            case Rees1963:
               electronRange = 4.57e-5 * pow(particle_energy[e], 1.75); // kg m^-2
               // Integrate downwards through the atmosphere to find density at depth=1
               for(int h=numAtmosphereLevels-1; h>=0; h--) {
                  if(atmosphere[h].depth / electronRange > 1) {
                     rho_R = atmosphere[h].density;
                     break;
                  }
               }
               if(rho_R == 0.) {
                  rho_R = atmosphere[0].density;
               }
               break;
            case Rees1989:
               // From Rees, M. H. (1989), q 3.4.4
               electronRange = 4.3e-6 + 5.36e-5 * pow(particle_energy[e], 1.67); // kg m^-2
               break;
            case SergienkoIvanov:
               electronRange = 1.64e-5 * pow(particle_energy[e], 1.67) * (1. + 9.48e-2 * pow(particle_energy[e], -1.57));
               break;
            case Robinson2020:
            case Juusola2025:
               // We don't need to actually do anything about the atmosphere here, and can just bail out.
               return;
            default:
               cerr << "(IONOSPHERE) Invalid value for Ionization model." << endl;
               abort();
         }

         for(int h=0; h<numAtmosphereLevels; h++) {
            Real lambda;
            Real rate=0;
            switch(ionizationModel) {
               case Rees1963:
                  // Rees et al 1963, eq. 1
                  lambda = ReesIsotropicLambda(atmosphere[h].depth/electronRange);
                  rate = particle_energy[e] / (electronRange / rho_R) / eps_ion_keV *   lambda   *   atmosphere[h].density / integratedDensity;
                  break;
               case Rees1989:
            // Rees 1989, eq. 3.3.7 / 3.3.8
                  lambda = ReesIsotropicLambda(atmosphere[h].depth/electronRange);
                  rate = particle_energy[e] * lambda * atmosphere[h].density / electronRange / eps_ion_keV;
                  break;
               case SergienkoIvanov:
                  lambda = SergienkoIvanovLambda(particle_energy[e]*1000., atmosphere[h].depth/electronRange);
                  rate = atmosphere[h].density / eps_ion_keV * particle_energy[e] * lambda / electronRange; // TODO: Albedo flux?
                  break;
               case Robinson2020:
               case Juusola2025:
                  // We don't need to actually do anything about the atmosphere here, and can just bail out.
                  return;
            }
            scatteringRate[e][h] = max(0., rate); // m^-1
         }
      }

      // Fill ionisation production table
      std::array< Real, SBC::productionNumParticleEnergies > differentialFlux; // Differential flux

      for(int e=0; e<productionNumAccEnergies; e++) {

         const Real productionAccEnergyStep = (log10(productionMaxAccEnergy) - log10(productionMinAccEnergy)) / productionNumAccEnergies;
         Real accenergy = pow(10., productionMinAccEnergy + e*(productionAccEnergyStep)); // In KeV

         for(int t=0; t<productionNumTemperatures; t++) {
            const Real productionTemperatureStep = (log10(productionMaxTemperature) - log10(productionMinTemperature)) / productionNumTemperatures;
            Real tempenergy = pow(10, productionMinTemperature + t*productionTemperatureStep); // In KeV

            for(int p=0; p<SBC::productionNumParticleEnergies; p++) {
               // TODO: Kappa distribution here? Now only going for maxwellian
               Real energyparam = (particle_energy[p]-accenergy)/tempenergy; // = E_p / (kB T)

               if(particle_energy[p] > accenergy) {
                  Real deltaE = (particle_energy[p+1] - particle_energy[p])* 1e3*physicalconstants::CHARGE;  // dE in J

                  differentialFlux[p] = sqrt(1. / (2. * M_PI * physicalconstants::MASS_ELECTRON))
                    * particle_energy[p] / tempenergy / sqrt(tempenergy * 1e3 *physicalconstants::CHARGE)
                    * deltaE * exp(-energyparam); // m / s  ... multiplied with density, this yields a flux 1/m^2/s
               } else {
                  differentialFlux[p] = 0;
               }
            }
            for(int h=0; h < numAtmosphereLevels; h++) {
               productionTable[h][e][t] = 0;
               for(int p=0; p<SBC::productionNumParticleEnergies; p++) {
                  productionTable[h][e][t] += scatteringRate[p][h]*differentialFlux[p];
               }
            }
         }
      }
   }

   /*!< Store the value of the magnetic field at the node.*/
   void SphericalTriGrid::storeNodeB() {
      for(uint n=0; n<nodes.size(); n++) {
         nodes[n].parameters[NODE_BX] = /*SBC::*/dipoleField(nodes[n].x[0],nodes[n].x[1],nodes[n].x[2],X,0,X) + /*SBC::*/BGB[0];
         nodes[n].parameters[NODE_BY] = /*SBC::*/dipoleField(nodes[n].x[0],nodes[n].x[1],nodes[n].x[2],Y,0,Y) + /*SBC::*/BGB[1];
         nodes[n].parameters[NODE_BZ] = /*SBC::*/dipoleField(nodes[n].x[0],nodes[n].x[1],nodes[n].x[2],Z,0,Z) + /*SBC::*/BGB[2];
      }
   }

   /* Look up the free electron production rate in the ionosphere, given the atmospheric height index,
    * particle energy after the ionospheric potential drop and inflowing distribution temperature */
   Real SphericalTriGrid::lookupProductionValue(int heightindex, Real energy_keV, Real temperature_keV) {
            Real normEnergy = (log10(energy_keV) - log10(productionMinAccEnergy)) / (log10(productionMaxAccEnergy) - log10(productionMinAccEnergy));
            if(normEnergy < 0) {
               normEnergy = 0;
            }
            Real normTemperature = (log10(temperature_keV) - log10(productionMinTemperature)) / (log(productionMaxTemperature) - log(productionMinTemperature));
            if(normTemperature < 0) {
               normTemperature = 0;
            }

            // Interpolation bin and parameters
            normEnergy *= productionNumAccEnergies;
            int energyindex = int(float(normEnergy));
            if(energyindex < 0) {
               energyindex = 0;
               normEnergy = 0;
            }
            if(energyindex > productionNumAccEnergies - 2) {
               energyindex = productionNumAccEnergies - 2;
               normEnergy = 0;
            }
            float t = normEnergy - floor(normEnergy);

            normTemperature *= productionNumTemperatures;
            int temperatureindex = int(float(normTemperature));
            float s = normTemperature - floor(normTemperature);
            if(temperatureindex < 0) {
               temperatureindex = 0;
               normTemperature = 0;
            }
            if(temperatureindex > productionNumTemperatures - 2) {
               temperatureindex = productionNumTemperatures - 2;
               normTemperature = 0;
            }

            // Lookup production rate by linearly interpolating table.
            return (productionTable[heightindex][energyindex][temperatureindex]*(1.-t) +
                    productionTable[heightindex][energyindex+1][temperatureindex] * t) * (1.-s) +
                   (productionTable[heightindex][energyindex][temperatureindex+1]*(1.-t) +
                    productionTable[heightindex][energyindex+1][temperatureindex+1] * t) * s ;

   }

   /* Estimate the magnetospheric electron precipitation energy flux (in W/m^2) from
    * mass density, electron temperature and potential difference.
    *
    * TODO: This is the coarse MHD estimate, lacking a better approximation. Should this
    * instead use the precipitation data reducer?
    */
   void SphericalTriGrid::calculatePrecipitation() {

      for(uint n=0; n<nodes.size(); n++) {
         Real ne = nodes[n].electronDensity();
         Real electronEnergy = nodes[n].electronTemperature() * physicalconstants::K_B;
         Real potential = nodes[n].deltaPhi();

         nodes[n].parameters[ionosphereParameters::PRECIP] = (ne / sqrt(2. * M_PI * physicalconstants::MASS_ELECTRON * electronEnergy))
            * (2. * electronEnergy * electronEnergy + 2 * physicalconstants::CHARGE * potential * electronEnergy
                  + (physicalconstants::CHARGE * potential)*(physicalconstants::CHARGE * potential));

      }
   }

   /* Calculate the conductivity tensor for every grid node, based on the
    * given F10.7 photospheric flux as a solar activity proxy.
    *
    * This assumes the FACs have already been coupled into the grid.
    *
    * If refillTensorAtRestart is true, we don't recompute precipitation and integration, we just refill the tensor from the sigmas as read from restart.
    * That is necessary so ig_inplanecurrent has non-zero data if an output file is written after restart and before the next ionosphere solution step.
    */
   void SphericalTriGrid::calculateConductivityTensor(
      const Real F10_7,
      const Real recombAlpha,
      const Real backgroundIonisation,
      const bool refillTensorAtRestart/*=false*/
   ) {
      phiprof::Timer timer {"ionosphere-calculateConductivityTensor"};

      // At restart we have SIGMAP, SIGMAH and SIGMAPARALLEL read in from the restart file already, no need to update here.
      if(!refillTensorAtRestart) {
         // Ranks that don't participate in ionosphere solving skip this function outright
         if(!isCouplingInwards && !isCouplingOutwards) {
            return;
         }

         calculatePrecipitation();

         if(ionosphereGrid.ionizationModel == Robinson2020) {

            // In the Robinson (2020) model, conductivity gets directly calculated from FACs.
            // DOI: doi/10.1029/2020JA028008
            const static std::array<Real,3> SigmaP0d_coefficients = {5., -0.8 , 60.9};
            const static std::array<Real,3> SigmaP0u_coefficients = {4.2, 1.1 ,318.6};
            const static std::array<Real,3> SigmaH0d_coefficients = {7.7, -1.8,139.0};
            const static std::array<Real,3> SigmaH0u_coefficients = {8.7, 4.6 ,327.1};

            const static std::array<Real,3> SigmaP1d_coefficients = {-3.2, -3.6, 21.9};
            const static std::array<Real,3> SigmaP1u_coefficients = { 6.8, -1.5,184.9};
            const static std::array<Real,3> SigmaH1d_coefficients = {-7.3,  5.6,100.9};
            const static std::array<Real,3> SigmaH1u_coefficients = {14.8,-10.4,129.9};

            // MLT interpolation (eq 7 from the paper)
            auto interpolate_robinson = [](const std::array<Real,3>& variable, Real MLT) -> Real {
               return variable[0] + variable[1] * cos(variable[2]/180.*M_PI + MLT);
            };

            // Smooth (cubic hermite) interpolation between two curves a and b, x is clamped to [-1; 1]
            auto smoothstep = [](Real a, Real b, Real x) -> Real {
               x = 0.5*(x+1);
               x = std::clamp((x-a)/(b-a),0.,1.);
               x = x*x*(3-2*x);
               return (1.-x)*a + x*b;
            };

            for(uint n=0; n<nodes.size(); n++) {

               Real MLT = atan2(nodes[n].x[1],nodes[n].x[0]);

               // Calculate FAC density through this node
               Real area = 0;
               for(uint e=0; e<nodes[n].numTouchingElements; e++) {
                  area += elementArea(nodes[n].touchingElements[e]);
               }
               area /= 3.; // As every element has 3 corners, don't double-count areas

               // The Robinson model wants FACS in microAmperes / m^2
               Real FAC = 1e6*nodes[n].parameters[ionosphereParameters::SOURCE]/area;

               // Get A, B and C factor by interpolation
               // Note: Positive FAC value -> downwards FACs.
               Real SigmaH0 = smoothstep(interpolate_robinson(SigmaH0u_coefficients, MLT), interpolate_robinson(SigmaH0d_coefficients, MLT), FAC/0.1);
               Real SigmaH1 = smoothstep(interpolate_robinson(SigmaH1u_coefficients, MLT), interpolate_robinson(SigmaH1d_coefficients, MLT), FAC/0.1);
               Real SigmaP0 = smoothstep(interpolate_robinson(SigmaP0u_coefficients, MLT), interpolate_robinson(SigmaP0d_coefficients, MLT), FAC/0.1);
               Real SigmaP1 = smoothstep(interpolate_robinson(SigmaP1u_coefficients, MLT), interpolate_robinson(SigmaP1d_coefficients, MLT), FAC/0.1);

               nodes[n].parameters[ionosphereParameters::SIGMAP] = SigmaP0 + SigmaP1 * FAC;
               nodes[n].parameters[ionosphereParameters::SIGMAH] = SigmaH0 + SigmaH1 * FAC;
               // TODO: What do we do about SIGMAPARALLEL?
            }
         } else if(ionosphereGrid.ionizationModel == Juusola2025) {

            const static int NODE_CONSTRAINT_REDUCTION = 1;
            const static int ELEMENT_CONSTRAINT_REDUCTION =1;

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
               MLT = fmod(MLT, 24.);
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
               MLT = fmod(MLT, 24.);
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
               MLT = fmod(MLT, 24.);
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
               MLT = fmod(MLT, 24.);
               int sector = MLT;
               Real interpolant = MLT - sector;
               return (1.-interpolant)*values[sector] + interpolant * values[(sector+1)%24];
            };

            // Eigen vector and matrix for solving
            // Each current-density vector lives in the circumcentre of each triangle
            // The constraints are calculated on each node
            Eigen::VectorXd vJ(2 * elements.size()); // 2 * elements.size() because we have two components of J in every element 
            Eigen::VectorXd vRHS1(nodes.size() + nodes.size()); // Right hand side for divergence-free system
            Eigen::VectorXd vRHS2(nodes.size() + nodes.size()); // Right hand side for curl-free system
            Eigen::SparseMatrix<Real> curlSolverMatrix(vRHS1.size(), vJ.size());

            std::vector<Real > elementCorrectionFactors(elements.size());

            // First, solve curl-free inplane current system.
            // Use those currents to estimate sigma ratio.
            // Then, solve divergence-free part.
            // Finally, estimate Sigmas.

            // This formalism uses a cirumcentre-based current-density vector field. 
            // 
            // To calculate the divergence at a specific node, the current density is
            // first interpolated to all the edges subtended by this node by weighing
            // the current-density of the elements subtended by a particular edge by
            // the proportion of the distances from the circumcentres to the midpoint
            // of that edge, to the line connecting the two circumcentres. This line
            // will always be the perpendicular bisector of the common edge, thanks to
            // the fact that circumcentres are equidistant from the corners of a
            // triangle. Then, the dot product of the edge-interpolated current
            // densities with the edge parallel is taken, multiplied by the length of
            // the dual to this edge, and summed over all edges subtended by the node. 
            // 
            // Since the mesh is not flat, the edge vectors are be transformed to a
            // common coordinate system (XY plane at the north pole) before the dot
            // product is taken 
            if(Eigen::loadMarket(curlSolverMatrix, "ionosphereSolverMatrix")){

               for(int n =0; n < nodes.size(); n++) {
                  vRHS1[n] = 0;
                  vRHS2[n] = nodes[n].parameters[ionosphereParameters::SOURCE];

               }

               for(int n = 0; n < nodes.size(); n++) {
                  vRHS1[n + nodes.size()] = ionosphereGrid.nodes[n].parameters[ionosphereParameters::SOURCE];
                  vRHS2[n + nodes.size()] = 0;

               }

         

            } else {
            // Divergence constraints
            for(uint gridNodeIndex=0; gridNodeIndex<nodes.size(); gridNodeIndex++) {

               // Divergence of divergence-free current
               vRHS1[gridNodeIndex] = 0;
               
               //Divergence of curl-free current
               vRHS2[gridNodeIndex] = nodes[gridNodeIndex].parameters[ionosphereParameters::SOURCE];

               for(uint32_t elLocalIndex=0; elLocalIndex<nodes[gridNodeIndex].numTouchingElements; elLocalIndex++) {
                  SphericalTriGrid::Element& element = elements[nodes[gridNodeIndex].touchingElements[elLocalIndex]];

                  // Find the two other nodes on this element 
                  int gridI=0,gridJ=0;
                  int localC=0,localI=0,localJ=0;
                  for(int c=0; c< 3; c++) {
                     if(element.corners[c] == gridNodeIndex) {
                        localC = c;
                        localI = (c+1)%3;
                        gridI=element.corners[localI];
                        localJ = (c+2)%3;
                        gridJ=element.corners[localJ];
                        break; 
                     }
                  }

                  int32_t otherElementi = findElementNeighbour(nodes[gridNodeIndex].touchingElements[elLocalIndex], localC, localI);
                  int32_t otherElementj = findElementNeighbour(nodes[gridNodeIndex].touchingElements[elLocalIndex], localC, localJ);

                  Eigen::Vector3d circumcentrem = elementCircumcentre(nodes[gridNodeIndex].touchingElements[elLocalIndex]);
                  Eigen::Vector3d midpointmi = commonEdgeMidpoint(nodes[gridNodeIndex].touchingElements[elLocalIndex], otherElementi);
                  Real li = (circumcentrem - midpointmi).norm();

                  Eigen::Vector3d rm(nodes[gridNodeIndex].x.data());
                  Eigen::Vector3d ri(nodes[gridI].x.data());
                  Eigen::Vector3d rj(nodes[gridJ].x.data());
                  Eigen::Vector3d edge = (ri - rm) / (ri - rm).norm();

                  Eigen::Vector3d normalm = elementNormal( nodes[gridNodeIndex].touchingElements[elLocalIndex]);
                  Eigen::Vector3d edgem = Eigen::Quaterniond::FromTwoVectors(normalm, Eigen::Vector3d::UnitZ()).toRotationMatrix() * edge;

                  curlSolverMatrix.coeffRef(gridNodeIndex, 2 * nodes[gridNodeIndex].touchingElements[elLocalIndex]) += edgem(0) * li;
                  curlSolverMatrix.coeffRef(gridNodeIndex, 2 * nodes[gridNodeIndex].touchingElements[elLocalIndex] + 1) += edgem(1) * li;

                  Eigen::Vector3d midpointmj = commonEdgeMidpoint( nodes[gridNodeIndex].touchingElements[elLocalIndex], otherElementj);
                  Real lj = (circumcentrem - midpointmj).norm();

                  edge = (rj - rm) / (rj - rm).norm();

                  edgem = Eigen::Quaterniond::FromTwoVectors(normalm, Eigen::Vector3d::UnitZ()).toRotationMatrix() * edge;

                  curlSolverMatrix.coeffRef(gridNodeIndex, 2 * nodes[gridNodeIndex].touchingElements[elLocalIndex]) += edgem(0) * lj;
                  curlSolverMatrix.coeffRef(gridNodeIndex, 2 * nodes[gridNodeIndex].touchingElements[elLocalIndex] + 1) += edgem(1) * lj;
                  
               }
            }
  
            // The curl at a specific node is calculated by taking half the dot product
            // of the edges opposite to the node with the current-density of the
            // elements subtended by the node, multiplied by a consistent orientation.
            // Curl constraints 
            for(uint n=0; n<nodes.size(); n++) {
               
               // Curl of divergence-free current
               vRHS1[nodes.size() + n] = nodes[n].parameters[ionosphereParameters::SOURCE];

               // Curl of curl-free current
               vRHS2[nodes.size() + n] = 0;
               
               for(uint32_t elLocalIndex=0; elLocalIndex<nodes[n].numTouchingElements; elLocalIndex++) {
                  SphericalTriGrid::Element& element = elements[nodes[n].touchingElements[elLocalIndex]];

                  // Find the two other nodes on this element 
                  int gridI=0,gridJ=0;
                  int localC=0,localI=0,localJ=0;
                  for(int c=0; c< 3; c++) {
                     if(element.corners[c] == n) {
                        localC = c;
                        localI = (c+1)%3;
                        gridI=element.corners[localI];
                        localJ = (c+2)%3;
                        gridJ=element.corners[localJ];
                        break; 
                     }
                  }

                  Eigen::Vector3d normal = elementNormal( nodes[n].touchingElements[elLocalIndex]);
                  Eigen::Vector3d ri(nodes[gridI].x.data());
                  Eigen::Vector3d rj(nodes[gridJ].x.data());
                  Eigen::Vector3d rm(nodes[n].x.data()); 

                  Eigen::Vector3d edgemi = (ri - rm) / (ri - rm).norm();
                  edgemi = Eigen::Quaterniond::FromTwoVectors(normal, Eigen::Vector3d::UnitZ()).toRotationMatrix() * edgemi;

                  Eigen::Vector3d edgemj = (rj - rm) / (rj - rm).norm();
                  edgemj = Eigen::Quaterniond::FromTwoVectors(normal, Eigen::Vector3d::UnitZ()).toRotationMatrix() * edgemj;

                  Real orientation = edgemj.cross(edgemi).dot(normal) > 0 ? 1. : -1.;

                  Eigen::Vector3d outerEdge = orientation * (rj - ri) / (rj - ri).norm();
                  Real outerEdgeLength = (rj - ri).norm();
                  outerEdge = Eigen::Quaterniond::FromTwoVectors(normal, Eigen::Vector3d::UnitZ()).toRotationMatrix() * outerEdge;

                  curlSolverMatrix.coeffRef(nodes.size() + n, 2 * nodes[n].touchingElements[elLocalIndex]) += outerEdge(0) * outerEdgeLength / 2.;
                  curlSolverMatrix.coeffRef(nodes.size() + n, 2 * nodes[n].touchingElements[elLocalIndex] + 1) += outerEdge(1) * outerEdgeLength / 2.;
               }


            }



            Eigen::saveMarket(curlSolverMatrix, "ionosphereSolverMatrix");

            }



            curlSolverMatrix.makeCompressed();
            
            // Solve curl-free currents.
            Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<Real>> solver;
            solver.compute(curlSolverMatrix);
            vJ = solver.solve(vRHS2);
            
            elementCurlFreeCurrent.resize(elements.size());
            elementDivFreeCurrent.resize(elements.size());
            for(uint el=0; el<elements.size(); el++) {
               std::array<uint32_t, 3>& corners = elements[el].corners;
               Real A = elementArea(el);
               Eigen::Vector3d r0(nodes[corners[0]].x.data());
               Eigen::Vector3d r1(nodes[corners[1]].x.data());
               Eigen::Vector3d r2(nodes[corners[2]].x.data());

               Eigen::Vector3d barycentre = (r0+r1+r2)/3.;

               Eigen::Vector3d rotatedVJ = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), barycentre.normalized()).toRotationMatrix() * Eigen::Vector3d(vJ[2*el], vJ[2*el+1], 0);
               elementCurlFreeCurrent[el] = rotatedVJ;

               Real MLT = atan2(barycentre[1], barycentre[0]) * 12 / M_PI + 12;

               // Note: The coefficients want to be looked up in A/km, so we multiply by 1000
               Real correction = pow(c4H(MLT)/c4P(MLT) * 1000*elementCurlFreeCurrent[el].norm(),1./(1.+c5P(MLT)-c5H(MLT))) / (1000*elementCurlFreeCurrent[el].norm());
               elementCorrectionFactors[el] = correction;
            } 

            // Apply correction to RHS for divergence-free current density (vRHS1)
            // Interpolate from elements to nodes via proportion of dual polygon contained
            for(uint n=0; n<nodes.size(); n++) {

               Real totalA = 0;
               Real correction = 0;

               for(uint32_t el=0; el< nodes[n].numTouchingElements; el++) {
                  Real A = areaInDualPolygon( n, nodes[n].touchingElements[el]);
                  totalA += A;
                  correction += elementCorrectionFactors[nodes[n].touchingElements[el]] * A;
               }
               correction /= totalA;

               //vRHS1[nodes.size()+n] = vRHS1[nodes.size()+n]*correction; 
            }  

            // Solve divergence-free system
            vJ = solver.solve(vRHS1);
            for(uint el=0; el<elements.size(); el++) {
               std::array<uint32_t, 3>& corners = elements[el].corners;
               Real A = elementArea(el);

               Eigen::Vector3d r0(nodes[corners[0]].x.data());
               Eigen::Vector3d r1(nodes[corners[1]].x.data());
               Eigen::Vector3d r2(nodes[corners[2]].x.data());

               Eigen::Vector3d barycentre = (r0+r1+r2)/3.;

               Eigen::Vector3d rotatedVJ = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), barycentre.normalized()).toRotationMatrix() * Eigen::Vector3d(vJ[2*el], vJ[2*el+1], 0);
               elementDivFreeCurrent[el] = rotatedVJ;
            }

            // Next, evaluate Sigma as a function of inplane-J and MLT
            #pragma omp parallel for
            for(uint n=0; n < nodes.size(); n++) {
               Eigen::Vector3d J{0,0,0};
               Eigen::Vector3d x(nodes[n].x.data());

               Real totalA=0;
               for(uint32_t el=0; el< nodes[n].numTouchingElements; el++) {
                  Real A = areaInDualPolygon( n, nodes[n].touchingElements[el]);
                  totalA += A;
                  J += elementDivFreeCurrent[nodes[n].touchingElements[el]] * A;
               }
               J/=totalA;

               Real MLT = atan2(x[1], x[0]) * 12 / M_PI + 12;

               // Formula 33 from Juusola et al 2025
               // (in A/km)
               J *= 1000;
               // cout << "J: " << J.norm() << endl;
               Real SigmaH = c4H(MLT) * pow(J.norm(), c5H(MLT));
               Real SigmaP = c4P(MLT) * pow(J.norm(), c5P(MLT));

               nodes[n].parameters[ionosphereParameters::SIGMAP] = SigmaP;
               nodes[n].parameters[ionosphereParameters::SIGMAH] = SigmaH;
            }

            
            // Perform distance transform on the mesh
            // Here we have, as temporary variables:
            // ZZPARAM -> index of closest node (so far)
            // PPARAM -> distance to boundary 
	         // cout << nodes[0].openFieldLine << endl;
            for(int n=0; n<nodes.size(); n++) {
               if(nodes[n].openFieldLine == FieldTracing::TracingLineEndType::CLOSED) {
                  nodes[n].parameters[ionosphereParameters::ZZPARAM] = n;
                  nodes[n].parameters[ionosphereParameters::PPARAM] = 0;
               } else {
                  nodes[n].parameters[ionosphereParameters::ZZPARAM] = -1;
                  nodes[n].parameters[ionosphereParameters::PPARAM] = 6371e3;
               }
            }

            bool done=false;
            while(!done) {
               done = true;
               for(int n=0; n<nodes.size(); n++) {
                  if(nodes[n].openFieldLine == FieldTracing::TracingLineEndType::CLOSED) {
                     continue; // Skip closed nodes
                  }
                  Eigen::Vector3d x(nodes[n].x.data());

                  for(int m=0; m<nodes[n].numTouchingElements; m++) {
                     SphericalTriGrid::Element& element = elements[nodes[n].touchingElements[m]];
                     for(int c=0; c<3; c++) {
                        int i = element.corners[c];
                        if(i == n) {
                           continue;
                        }

                        if(nodes[i].openFieldLine  == FieldTracing::TracingLineEndType::CLOSED) {
                           // Closed nodes can be probed directly
                           Eigen::Vector3d ox(nodes[i].x.data());
                           Real distance = (ox - x).norm();
                           if(distance < nodes[n].parameters[ionosphereParameters::PPARAM]) {
                              nodes[n].parameters[ionosphereParameters::PPARAM] = distance;
                              nodes[n].parameters[ionosphereParameters::ZZPARAM] = i;
                              done = false;
                           }
                        } else {
                           // Open nodes require inferred distance
                           // TODO: This should actually be geodetic distance, but maybe we can afford not to care
                           if(nodes[i].parameters[ionosphereParameters::ZZPARAM] == -1) {
                              // This node doesn't even have a distance yet, skipping.
                              //done = false;
                              continue;
                           }

                           Eigen::Vector3d ox(nodes[ nodes[i].parameters[ionosphereParameters::ZZPARAM] ].x.data());
                           Real distance = (ox - x).norm();
                           if(distance < nodes[n].parameters[ionosphereParameters::PPARAM]) {
                              nodes[n].parameters[ionosphereParameters::PPARAM] = distance;
                              nodes[n].parameters[ionosphereParameters::ZZPARAM] = nodes[i].parameters[ionosphereParameters::ZZPARAM];
                              done = false;
                           }
                        }
                     }
                  }
               }
            }

            #pragma omp parallel for
            for(int n=0; n<nodes.size(); n++) {

               // Adjust sigmas based on distance value
               if(nodes[n].parameters[ionosphereParameters::PPARAM] > 300e3) { // TODO: Hardcoded 300km here
                  Real alpha = (nodes[n].parameters[ionosphereParameters::PPARAM] - 300e3) / 300e3;
                  nodes[n].parameters[ionosphereParameters::SIGMAP] *= exp(-alpha);
                  nodes[n].parameters[ionosphereParameters::SIGMAH] *= exp(-alpha);
               }
				
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


               // Drop-in replacement for cosine function for describing plasma
               // production at the height of max plasma production using the
               // Chapman function (which assumes the earth is round, not flat).
               //
               // The advantage of this approach is that the conductance gradient at the terminator is
               // more realistic. This is important since conductance gradients appear in the equations that
               // relate electric and magnetic fields. In addition, conductances above 90° sza are positive.
               // The code is based on table lookup, and does not calculate the Chapman function.
               // Author: S. M. Hatch (2024)
               auto altcos = [](Real sza)->Real {
                  Real degrees = fabs(sza) / M_PI * 180;

                  // Clamp to table lookup range
                  degrees = max(0.,degrees);
                  degrees = min(120.,degrees);

                  int bin = degrees * 10.;
                  Real interpolant = bin - (degrees * 10.);
                  return (1.-interpolant) * chapman_euv_table[bin] + interpolant * chapman_euv_table[bin+1];
               };


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
               b.normalized();
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
            // The other (atmospheric precipitation and height-integration-based) models
            // share most of their code

            //Calculate height-integrated conductivities and 3D electron density
            // TODO: effdt > 0?
            // (Then, ne += dt*(q - alpha*ne*abs(ne))
            for(uint n=0; n<nodes.size(); n++) {
               nodes[n].parameters[ionosphereParameters::SIGMAP] = 0;
               nodes[n].parameters[ionosphereParameters::SIGMAH] = 0;
               nodes[n].parameters[ionosphereParameters::SIGMAPARALLEL] = 0;
               std::array<Real, numAtmosphereLevels> electronDensity;

               // Note this loop counts from 1 (std::vector is zero-initialized, so electronDensity[0] = 0)
               for(int h=1; h<numAtmosphereLevels; h++) {
                  // Calculate production rate
                  Real energy_keV = max(nodes[n].deltaPhi()/1000., productionMinAccEnergy);

                  Real ne = nodes[n].electronDensity();
                  Real electronTemp = nodes[n].electronTemperature();
                  Real temperature_keV = (physicalconstants::K_B / physicalconstants::CHARGE) / 1000. * electronTemp;
                  if(!(std::isfinite(energy_keV) && std::isfinite(temperature_keV))) {
                     cerr << "(ionosphere) NaN or inf encountered in conductivity calculation: " << endl
                        << "   `-> DeltaPhi     = " << nodes[n].deltaPhi()/1000. << " keV" << endl
                        << "   `-> energy_keV   = " << energy_keV << endl
                        << "   `-> ne           = " << ne << " m^-3" << endl
                        << "   `-> electronTemp = " << electronTemp << " K" << endl;
                  }
                  Real qref = ne * lookupProductionValue(h, energy_keV, temperature_keV);

                  // Get equilibrium electron density
                  electronDensity[h] = sqrt(qref/recombAlpha);

                  // Calculate conductivities
                  Real halfdx = 1000 * 0.5 * (atmosphere[h].altitude -  atmosphere[h-1].altitude);
                  Real halfCH = halfdx * 0.5 * (atmosphere[h-1].hallcoeff + atmosphere[h].hallcoeff);
                  Real halfCP = halfdx * 0.5 * (atmosphere[h-1].pedersencoeff + atmosphere[h].pedersencoeff);
                  Real halfCpara = halfdx * 0.5 * (atmosphere[h-1].parallelcoeff + atmosphere[h].parallelcoeff);

                  nodes[n].parameters[ionosphereParameters::SIGMAP] += (electronDensity[h]+electronDensity[h-1]) * halfCP;
                  nodes[n].parameters[ionosphereParameters::SIGMAH] += (electronDensity[h]+electronDensity[h-1]) * halfCH;
                  nodes[n].parameters[ionosphereParameters::SIGMAPARALLEL] += (electronDensity[h]+electronDensity[h-1]) * halfCpara;
               }
            }
         }
      }

      // Antisymmetric tensor epsilon_ijk
      static const int epsilon[3][3][3] = {
         {{0,0,0},{0,0,1},{0,-1,0}},
         {{0,0,-1},{0,0,0},{1,0,0}},
         {{0,1,0},{-1,0,0},{0,0,0}}
      };


      for(uint n=0; n<nodes.size(); n++) {

         std::array<Real, 3>& x = nodes[n].x;
         // TODO: Perform coordinate transformation here?

         // At restart we have SIGMAP, SIGMAH and SIGMAPARALLEL read in from the restart file already.
         if(!refillTensorAtRestart) {

#ifdef MOEN_AND_BREKKE
            // Solar incidence parameter for calculating UV ionisation on the dayside
            Real coschi = x[0] / Ionosphere::innerRadius;
            if(coschi < 0) {
               coschi = 0;
            }

            // Pre-transformed F10_7 values
            Real F10_7_p_049 = pow(F10_7, 0.49);
            Real F10_7_p_053 = pow(F10_7, 0.53);
            Real sigmaP_dayside = backgroundIonisation + F10_7_p_049 * (0.34 * coschi + 0.93 * sqrt(coschi));
            Real sigmaH_dayside = backgroundIonisation + F10_7_p_053 * (0.81 * coschi + 0.54 * sqrt(coschi));
#else // Not MOEN_AND_BREKKE, but JUUSOLA2025
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

            // Drop-in replacement for cosine function for describing plasma
            // production at the height of max plasma production using the
            // Chapman function (which assumes the earth is round, not flat).
            //
            // The advantage of this approach is that the conductance gradient at the terminator is
            // more realistic. This is important since conductance gradients appear in the equations that
            // relate electric and magnetic fields. In addition, conductances above 90° sza are positive.
            // The code is based on table lookup, and does not calculate the Chapman function.
            // Author: S. M. Hatch (2024)
            auto altcos = [](Real sza)->Real {
               Real degrees = fabs(sza) / M_PI * 180;

               // Clamp to table lookup range
               degrees = max(0.,degrees);
               degrees = min(120.,degrees);

               int bin = degrees * 10.;
               Real interpolant = bin - (degrees * 10.);
               return (1.-interpolant) * chapman_euv_table[bin] + interpolant * chapman_euv_table[bin+1];
            };

            // Solar incidence parameter for calculating UV ionisation on the dayside
            Real coschi = x[0] / Ionosphere::innerRadius;
            Real chi = acos(coschi);
            Real qprime = altcos(chi);

            const Real c1p = 0.351;
            const Real c2p = 0.697;
            const Real c3p = 0.707;
            const Real c1h = 0.720;
            const Real c2h = 0.617;
            const Real c3h = 0.846;
            Real sigmaP_dayside = c1p * pow(F10_7, c2p) * pow(qprime, c3p);
            Real sigmaH_dayside = c1h * pow(F10_7, c2h) * pow(qprime, c3h);
#endif

            nodes[n].parameters[ionosphereParameters::SIGMAP] = sqrt( pow(nodes[n].parameters[ionosphereParameters::SIGMAP],2) + pow(sigmaP_dayside,2));
            nodes[n].parameters[ionosphereParameters::SIGMAH] = sqrt( pow(nodes[n].parameters[ionosphereParameters::SIGMAH],2) + pow(sigmaH_dayside,2));
         }

         // Build conductivity tensor
         Real sigmaP = nodes[n].parameters[ionosphereParameters::SIGMAP];
         Real sigmaH = nodes[n].parameters[ionosphereParameters::SIGMAH];
         Real sigmaParallel = nodes[n].parameters[ionosphereParameters::SIGMAPARALLEL];

         // GUMICS-Style conductivity tensor.
         // Approximate B vector = radial vector
         // SigmaP and SigmaH are both in-plane with the mesh
         // No longitudinal conductivity
         if(Ionosphere::conductivityModel == Ionosphere::GUMICS) {
            std::array<Real, 3> b = {x[0] / Ionosphere::innerRadius, x[1] / Ionosphere::innerRadius, x[2] / Ionosphere::innerRadius};
            if(x[2] >= 0) {
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
         } else if(Ionosphere::conductivityModel == Ionosphere::Ridley) {

            sigmaParallel = Ionosphere::ridleyParallelConductivity;
            std::array<Real, 3> b = {
               dipoleField(x[0],x[1],x[2],X,0,X),
               dipoleField(x[0],x[1],x[2],Y,0,Y),
               dipoleField(x[0],x[1],x[2],Z,0,Z)
            };
            Real Bnorm = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
            b[0] /= Bnorm;
            b[1] /= Bnorm;
            b[2] /= Bnorm;

            for(int i=0; i<3; i++) {
               for(int j=0; j<3; j++) {
                  nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] = sigmaP * ((i==j)? 1. : 0.) + (sigmaParallel - sigmaP)*b[i]*b[j];
                  for(int k=0; k<3; k++) {
                     nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] -= sigmaH * epsilon[i][j][k]*b[k];
                  }
               }
            }
         } else if(Ionosphere::conductivityModel == Ionosphere::Koskinen) {

            std::array<Real, 3> b = {
               dipoleField(x[0],x[1],x[2],X,0,X),
               dipoleField(x[0],x[1],x[2],Y,0,Y),
               dipoleField(x[0],x[1],x[2],Z,0,Z)
            };
            Real Bnorm = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
            b[0] /= Bnorm;
            b[1] /= Bnorm;
            b[2] /= Bnorm;

            for(int i=0; i<3; i++) {
               for(int j=0; j<3; j++) {
                  nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] = sigmaP * ((i==j)? 1. : 0.) + (sigmaParallel - sigmaP)*b[i]*b[j];
                  for(int k=0; k<3; k++) {
                     nodes[n].parameters[ionosphereParameters::SIGMA + i*3 + j] -= sigmaH * epsilon[i][j][k]*b[k];
                  }
               }
            }
         } else {
            cerr << "(ionosphere) Error: Undefined conductivity model " << Ionosphere::conductivityModel << "! Ionospheric Sigma Tensor will be zero." << endl;
         }
      }
   }









   // (Re-)create the subcommunicator for ionosphere-internal communication
   // This needs to be rerun after Vlasov grid load balancing to ensure that
   // ionosphere info is still communicated to the right ranks.
   void SphericalTriGrid::updateIonosphereCommunicator(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid, FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid) {
      phiprof::Timer timer {"ionosphere-updateIonosphereCommunicator"};

      // Check if the current rank contains ionosphere boundary cells.
      isCouplingOutwards = true;
      //for(const auto& cell: mpiGrid.get_cells()) {
      //   if(mpiGrid[cell]->sysBoundaryFlag == sysboundarytype::IONOSPHERE) {
      //      isCouplingOutwards = true;
      //   }
      //}

      // If a previous communicator existed, destroy it.
      if(communicator != MPI_COMM_NULL) {
         MPI_Comm_free(&communicator);
         communicator = MPI_COMM_NULL;
      }

      // Whether or not the current rank is coupling inwards from fsgrid was determined at
      // grid initialization time and does not change during runtime.
      int writingRankInput=0;
      if(isCouplingInwards || isCouplingOutwards) {
         int size;
         MPI_Comm_split(MPI_COMM_WORLD, 1, technicalGrid.getRank(), &communicator);
         MPI_Comm_rank(communicator, &rank);
         MPI_Comm_size(communicator, &size);
         if(rank == 0) {
            writingRankInput = technicalGrid.getRank();
         }

      } else {
         MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &communicator); // All other ranks are staying out of the communicator.
         rank = -1;
      }

      // Make sure all tasks know which task on MPI_COMM_WORLD does the writing
      MPI_Allreduce(&writingRankInput, &writingRank, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
   }

   // Calculate upmapped potential at the given coordinates,
   // by tracing down to the ionosphere and interpolating the appropriate element
   Real SphericalTriGrid::interpolateUpmappedPotential(
      const std::array<Real, 3>& x
   ) {

      if(!this->dipoleField) {
         // Timestep zero => apparently the dipole field is not initialized yet.
         return 0.;
      }
      Real potential = 0;

      // Do we have a stored coupling for these coordinates already?
      #pragma omp critical(coupling)
      {
         if(vlasovGridCoupling.find(x) == vlasovGridCoupling.end()) {

            // If not, create one.
            vlasovGridCoupling[x] = FieldTracing::calculateIonosphereVlasovGridCoupling(x, nodes, Ionosphere::radius);
         }

         const std::array<std::pair<int, Real>, 3>& coupling = vlasovGridCoupling[x];

         for(int i=0; i<3; i++) {
            potential += coupling[i].second * nodes[coupling[i].first].parameters[ionosphereParameters::SOLUTION];
         }
      }
      return potential;
   }

   // Transport field-aligned currents down from the simulation cells to the ionosphere
   void SphericalTriGrid::mapDownBoundaryData(
       FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
       FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
       FsGrid< std::array<Real, fsgrids::moments::N_MOMENTS>, FS_STENCIL_WIDTH> & momentsGrid,
         FsGrid< std::array<Real, fsgrids::volfields::N_VOL>, FS_STENCIL_WIDTH> & volGrid,
       FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid) {

      if(!isCouplingInwards && !isCouplingOutwards) {
         return;
      }

      phiprof::Timer timer {"ionosphere-mapDownMagnetosphere"};

      // Create zeroed-out input arrays
      std::vector<double> FACinput(nodes.size());
      std::vector<double> rhoInput(nodes.size());
      std::vector<double> temperatureInput(nodes.size());

      // Map all coupled nodes down into it
      // Tasks that don't have anything to couple to can skip this step.
      if(isCouplingInwards) {
      #pragma omp parallel for
         for(uint n=0; n<nodes.size(); n++) {

            Real nodeAreaGeometric = 0;

            // Map down FAC based on magnetosphere rotB
            if(nodes[n].xMapped[0] == 0. && nodes[n].xMapped[1] == 0. && nodes[n].xMapped[2] == 0.) {
               // Skip cells that couple nowhere
               continue;
            }

            // Local cell
            std::array<FsGridTools::FsIndex_t,3> lfsc = getLocalFsGridCellIndexForCoord(technicalGrid,nodes[n].xMapped);
            if(lfsc[0] == -1 || lfsc[1] == -1 || lfsc[2] == -1) {
               continue;
            }

            // Iterate through the elements touching that node
            for(uint e=0; e<nodes[n].numTouchingElements; e++) {
               // Also sum up touching elements' areas and upmapped areas to compress
               // density and temperature with them
               // TODO: Precalculate this?
               nodeAreaGeometric += elementArea(nodes[n].touchingElements[e]);
            }

            // Divide by 3, as every element will be counted from each of its
            // corners.  Prevent areas from being multiply-counted
            nodeAreaGeometric /= 3.;

            // Calc curlB, note division by DX one line down
            const std::array<Real, 3> curlB = interpolateCurlB(
               perBGrid,
               dPerBGrid,
               technicalGrid,
               FieldTracing::fieldTracingParameters.reconstructionCoefficientsCache,
               lfsc[0],lfsc[1],lfsc[2],
               nodes[n].xMapped
            );

            // Dot curl(B) with normalized B, scale by ratio of B(ionosphere)/B(upmapped), multiply by geometric area around ionosphere node to obtain current from density
            FACinput[n] = nodeAreaGeometric * (nodes[n].parameters[ionosphereParameters::UPMAPPED_BX]*curlB[0] + nodes[n].parameters[ionosphereParameters::UPMAPPED_BY]*curlB[1] + nodes[n].parameters[ionosphereParameters::UPMAPPED_BZ]*curlB[2])
               * sqrt((nodes[n].parameters[ionosphereParameters::NODE_BX]*nodes[n].parameters[ionosphereParameters::NODE_BX]
                  + nodes[n].parameters[ionosphereParameters::NODE_BY]*nodes[n].parameters[ionosphereParameters::NODE_BY]
                  + nodes[n].parameters[ionosphereParameters::NODE_BZ]*nodes[n].parameters[ionosphereParameters::NODE_BZ])
               )
               / ((nodes[n].parameters[ionosphereParameters::UPMAPPED_BX]*nodes[n].parameters[ionosphereParameters::UPMAPPED_BX]
                  + nodes[n].parameters[ionosphereParameters::UPMAPPED_BY]*nodes[n].parameters[ionosphereParameters::UPMAPPED_BY]
                  + nodes[n].parameters[ionosphereParameters::UPMAPPED_BZ]*nodes[n].parameters[ionosphereParameters::UPMAPPED_BZ])
               * physicalconstants::MU_0 * technicalGrid.DX
            );

            // By definition, a downwards current into the ionosphere has a positive FAC value,
            // as it corresponds to positive divergence of horizontal current in the ionospheric plane.
            // To make sure we match that, flip FAC sign on the southern hemisphere
            if(nodes[n].x[2] < 0) {
               FACinput[n] *= -1;
            }

            std::array<Real,3> frac = getFractionalFsGridCellForCoord(technicalGrid,nodes[n].xMapped);
            for(int c=0; c<3; c++) {
               // Shift by half a cell, as we are sampling volume quantities that are logically located at cell centres.
               if(frac[c] < 0.5) {
                  lfsc[c] -= 1;
                  frac[c] += 0.5;
               } else {
                  frac[c] -= 0.5;
               }
            }

            // Linearly interpolate neighbourhood
            Real couplingSum = 0;
            for(int xoffset : {0,1}) {
               for(int yoffset : {0,1}) {
                  for(int zoffset : {0,1}) {

                     Real coupling = (1. - abs(xoffset - frac[0])) * (1. - abs(yoffset - frac[1])) * (1. - abs(zoffset - frac[2]));
                     if(coupling < 0. || coupling > 1.) {
                        cerr << "Ionosphere warning: node << " << n << " has coupling value " << coupling <<
                           ", which is outside [0,1] at line " << __LINE__ << "!" << endl;
                     }

                     // Only couple to actual simulation cells
                     if(technicalGrid.get(lfsc[0]+xoffset,lfsc[1]+yoffset,lfsc[2]+zoffset)->sysBoundaryFlag == sysboundarytype::NOT_SYSBOUNDARY) {
                        couplingSum += coupling;
                     } else {
                        continue;
                     }


                     // Map density, temperature down
                     Real thisCellRho = momentsGrid.get(lfsc[0]+xoffset,lfsc[1]+yoffset,lfsc[2]+zoffset)->at(fsgrids::RHOQ) / physicalconstants::CHARGE;
                     rhoInput[n] += coupling * thisCellRho;
                     temperatureInput[n] += coupling * 1./3. * (
                        momentsGrid.get(lfsc[0]+xoffset,lfsc[1]+yoffset,lfsc[2]+zoffset)->at(fsgrids::P_11) +
                        momentsGrid.get(lfsc[0]+xoffset,lfsc[1]+yoffset,lfsc[2]+zoffset)->at(fsgrids::P_22) +
                        momentsGrid.get(lfsc[0]+xoffset,lfsc[1]+yoffset,lfsc[2]+zoffset)->at(fsgrids::P_33)) / (thisCellRho * physicalconstants::K_B * ion_electron_T_ratio);
                  }
               }
            }

            // The coupling values *would* have summed to 1 in free and open space, but since we are close to the inner
            // boundary, some cells were skipped, as they are in the sysbondary. Renormalize values by dividing by the couplingSum.
            if(couplingSum > 0) {
               rhoInput[n] /= couplingSum;
               temperatureInput[n] /= couplingSum;
            }
         }
      }

      // Allreduce on the ionosphere communicator
      std::vector<double> FACsum(nodes.size());
      std::vector<double> rhoSum(nodes.size());
      std::vector<double> temperatureSum(nodes.size());
      MPI_Allreduce(&FACinput[0], &FACsum[0], nodes.size(), MPI_DOUBLE, MPI_SUM, communicator);
      MPI_Allreduce(&rhoInput[0], &rhoSum[0], nodes.size(), MPI_DOUBLE, MPI_SUM, communicator);
      MPI_Allreduce(&temperatureInput[0], &temperatureSum[0], nodes.size(), MPI_DOUBLE, MPI_SUM, communicator); // TODO: Does it make sense to SUM the temperatures?

      for(uint n=0; n<nodes.size(); n++) {

         // Adjust densities by the loss-cone filling factor.
         // This is an empirical smooothstep function that artificially reduces
         // downmapped density below auroral latitudes.
         Real theta = acos(nodes[n].x[2] / sqrt(nodes[n].x[0]*nodes[n].x[0] + nodes[n].x[1]*nodes[n].x[1] + nodes[n].x[2]*nodes[n].x[2])); // Latitude
         if(theta > M_PI/2.) {
            theta = M_PI - theta;
         }
         // Smoothstep with an edge at about 67 deg.
         Real Chi0 = 0.01 + 0.99 * .5 * (1 + tanh((23. - theta * (180. / M_PI)) / 6));

         if(rhoSum[n] == 0 || temperatureSum[n] == 0) {
            // Node couples nowhere. Assume some default values.
            nodes[n].parameters[ionosphereParameters::SOURCE] = 0;

            nodes[n].parameters[ionosphereParameters::RHON] = Ionosphere::unmappedNodeRho * Chi0;
            nodes[n].parameters[ionosphereParameters::TEMPERATURE] = Ionosphere::unmappedNodeTe;
         } else {
            // Store as the node's parameter values.
            if(Ionosphere::couplingTimescale == 0) {
               // Immediate coupling
               nodes[n].parameters[ionosphereParameters::SOURCE] = FACsum[n];
               nodes[n].parameters[ionosphereParameters::RHON] = rhoSum[n] * Chi0;
               nodes[n].parameters[ionosphereParameters::TEMPERATURE] = temperatureSum[n];
            } else {

               // Slow coupling with a given timescale.
               // See https://en.wikipedia.org/wiki/Exponential_smoothing#Time_constant
               // P::dt valid for shorter coupling periods or Ionosphere::couplingInterval == 0 meaning every step
               Real timeInterval = Parameters::dt;
               if(Ionosphere::couplingInterval > Parameters::dt) {
                  timeInterval = Ionosphere::couplingInterval;
               }
               Real a = 1. - exp(- timeInterval / Ionosphere::couplingTimescale);
               if(a>1) {
                  a=1.;
               }

               nodes[n].parameters[ionosphereParameters::SOURCE] = (1.-a) * nodes[n].parameters[ionosphereParameters::SOURCE] + a * FACsum[n];
               nodes[n].parameters[ionosphereParameters::RHON] = (1.-a) * nodes[n].parameters[ionosphereParameters::RHON] + a * rhoSum[n] * Chi0;
               nodes[n].parameters[ionosphereParameters::TEMPERATURE] = (1.-a) * nodes[n].parameters[ionosphereParameters::TEMPERATURE] + a * temperatureSum[n];
            }
         }

      }

      // Make sure FACs are balanced, so that the potential doesn't start to drift
      offset_FAC();

   }

   // Calculate grad(T) for a element basis function that is zero at corners a and b,
   // and unity at corner c
   std::array<Real,3> SphericalTriGrid::computeGradT(const std::array<Real, 3>& a,
                                       const std::array<Real, 3>& b,
                                       const std::array<Real, 3>& c) {

     Vec3d av(a[0],a[1],a[2]);
     Vec3d bv(b[0],b[1],b[2]);
     Vec3d cv(c[0],c[1],c[2]);

     Vec3d z = cross_product(bv-cv, av-cv);

     Vec3d result = cross_product(z,bv-av)/dot_product( z, cross_product(av,bv) + cross_product(cv, av-bv));

     return std::array<Real,3>{result[0],result[1],result[2]};
   }

   // Calculate the average sigma tensor of an element by averaging over the three nodes it touches
   std::array<Real, 9> SphericalTriGrid::sigmaAverage(uint elementIndex) {

     std::array<Real, 9> retval{0,0,0,0,0,0,0,0,0};

     for(int corner=0; corner<3; corner++) {
       Node& n = nodes[ elements[elementIndex].corners[corner] ];
       for(int i=0; i<9; i++) {
         retval[i] += n.parameters[ionosphereParameters::SIGMA + i] / 3.;
       }
     }

     return retval;
   }

   // calculate integral( grd(Ti) Sigma grad(Tj) ) over the area of the given element
   // The i and j parameters enumerate the piecewise linear element basis function
   Real SphericalTriGrid::elementIntegral(uint elementIndex, int i, int j, bool transpose) {

     Element& e = elements[elementIndex];
     const std::array<Real, 3>& c1 = nodes[e.corners[0]].x;
     const std::array<Real, 3>& c2 = nodes[e.corners[1]].x;
     const std::array<Real, 3>& c3 = nodes[e.corners[2]].x;

     std::array<Real, 3> Ti,Tj;
     switch(i) {
     case 0:
       Ti = computeGradT(c2,c3,c1);
       break;
     case 1:
       Ti = computeGradT(c1,c3,c2);
       break;
     case 2: default:
       Ti = computeGradT(c1,c2,c3);
       break;
     }
     switch(j) {
     case 0:
       Tj = computeGradT(c2,c3,c1);
       break;
     case 1:
       Tj = computeGradT(c1,c3,c2);
       break;
     case 2: default:
       Tj = computeGradT(c1,c2,c3);
       break;
     }

     std::array<Real, 9> sigma = sigmaAverage(elementIndex);

     Real retval = 0;
     if(transpose) {
       for(int n=0; n<3; n++) {
         for(int m=0; m<3; m++) {
           retval += Ti[m] * sigma[3*n+m] * Tj[n];
         }
       }
     } else {
       for(int n=0; n<3; n++) {
         for(int m=0; m<3; m++) {
           retval += Ti[n] * sigma[3*n+m] * Tj[m];
         }
       }
     }

     return retval * elementArea(elementIndex);
   }

   // Add matrix value for the solver, linking two nodes.
   void SphericalTriGrid::addMatrixDependency(uint node1, uint node2, Real coeff, bool transposed) {

     // No need to bother with zero coupling
     if(coeff == 0) {
       return;
     }

     // Special case handling for Gauge fixing. Gauge-fixed nodes only couple to themselves.
     if(ionosphereGrid.gaugeFixing == Pole) {
       if( (!transposed && node1 == 0) ||
           (transposed && node2 == 0)) {
         if(node1 == node2) {
            coeff = 1;
         } else {
            return;
         }
       }
     } else if(ionosphereGrid.gaugeFixing == Equator) {
        if( (!transposed && fabs(nodes[node1].x[2]) < Ionosphere::innerRadius * sin(Ionosphere::shieldingLatitude * M_PI / 180.0)) ||
            ( transposed && fabs(nodes[node2].x[2]) < Ionosphere::innerRadius * sin(Ionosphere::shieldingLatitude * M_PI / 180.0))) {
           if(node1 == node2) {
              coeff = 1;
           } else {
              return;
           }
        }
     }

     Node& n = nodes[node1];
     // First check if the dependency already exists
     for(uint i=0; i<n.numDepNodes; i++) {
       if(n.dependingNodes[i] == node2) {

         // Yup, found it, let's simply add the coefficient.
         if(transposed) {
           n.transposedCoeffs[i] += coeff;
         } else {
           n.dependingCoeffs[i] += coeff;
         }
         return;
       }
     }

     // Not found, let's add it.
     if(n.numDepNodes >= MAX_DEPENDING_NODES-1) {
       // This shouldn't happen (but did in tests!)
       cerr << "(ionosphere) Node " << node1 << " already has " << MAX_DEPENDING_NODES << " depending nodes:" << endl;
       cerr << "     [ ";
       for(int i=0; i< MAX_DEPENDING_NODES; i++) {
         cerr << n.dependingNodes[i] << ", ";
       }
       cerr << " ]." << endl;

       std::set<uint> neighbourNodes;
       for(uint e = 0; e<nodes[node1].numTouchingElements; e++) {
         Element& E = elements[nodes[node1].touchingElements[e]];
         for(int c=0; c<3; c++) {
           neighbourNodes.emplace(E.corners[c]);
         }
       }
       cerr << "    (it has " << nodes[node1].numTouchingElements << " neighbour elements and "
         << neighbourNodes.size()-1 << " direct neighbour nodes:" << endl << "    [ ";
       for(auto& n : neighbourNodes) {
          if(n != node1) {
            cerr << n << ", ";
          }
       }
       cerr << "])." << endl;
       return;
     }
     n.dependingNodes[n.numDepNodes] = node2;
     if(transposed) {
       n.dependingCoeffs[n.numDepNodes] = 0;
       n.transposedCoeffs[n.numDepNodes] = coeff;
     } else {
       n.dependingCoeffs[n.numDepNodes] = coeff;
       n.transposedCoeffs[n.numDepNodes] = 0;
     }
     n.numDepNodes++;
   }

   // Add solver matrix dependencies for the neighbouring nodes
   void SphericalTriGrid::addAllMatrixDependencies(uint nodeIndex) {

     nodes[nodeIndex].numDepNodes = 1;

     // Add selfcoupling dependency already, to guarantee that it sits at index 0
     nodes[nodeIndex].dependingNodes[0] = nodeIndex;
     nodes[nodeIndex].dependingCoeffs[0] = 0;
     nodes[nodeIndex].transposedCoeffs[0] = 0;

     for(uint t=0; t<nodes[nodeIndex].numTouchingElements; t++) {
       int j0=-1;
       Element& e = elements[nodes[nodeIndex].touchingElements[t]];

       // Find the corner this node is touching
       for(int c=0; c <3; c++) {
         if(e.corners[c] == nodeIndex) {
           j0=c;
         }
       }

       // Special case: we are touching the middle of an edge
       if(j0 == -1) {
          // This is not implemented. Instead, refinement interfaces are stitched, so these kinds
          // of T-junctions never appear.
       } else {

         // Normal case.
         for(int c=0; c <3; c++) {
           uint neigh=e.corners[c];
           addMatrixDependency(nodeIndex, neigh, elementIntegral(nodes[nodeIndex].touchingElements[t], j0, c));
           addMatrixDependency(nodeIndex, neigh, elementIntegral(nodes[nodeIndex].touchingElements[t], j0, c,true),true);
         }
       }
     }
   }

   // Make sure refinement interfaces are properly "stitched", and that there are no
   // nodes remaining on t-junctions. This is done by splitting the bigger neighbour:
   //
   //      A---------------C         A---------------C
   //     / \             /         / \  resized .-'/
   //    /   \           /         /   \      .-'  /
   //   /     \         /         /     \  .-'    /
   //  o-------n       /    ==>  o-------n' new  /.  <- potential other node to update next?
   //   \     / \     /           \     / \     /  .
   //    \   /   \   /             \   /   \   /    .
   //     \ /     \ /               \ /     \ /      .
   //      o-------B                 o-------B . . .  .
   void SphericalTriGrid::stitchRefinementInterfaces() {

      for(uint n=0; n<nodes.size(); n++) {

         for(uint t=0; t<nodes[n].numTouchingElements; t++) {
            Element& e = elements[nodes[n].touchingElements[t]];
            int j0=-1;

            // Find the corner this node is touching
            for(int c=0; c <3; c++) {
               if(e.corners[c] == n) {
                  j0=c;
               }
            }

            if(j0 != -1) {
               // Normal element corner
               continue;
            }

            // Not a corner of this element => Split element

            // Find the corners of this element that we are collinear with
            uint A=0,B=0,C=0;
            Real bestColinearity = 0;
            for(int c=0; c <3; c++) {
               Node& a=nodes[e.corners[c]];
               Node& b=nodes[e.corners[(c+1)%3]];
               Vec3d ab(b.x[0] - a.x[0], b.x[1] - a.x[1], b.x[2] - a.x[2]);
               Vec3d an(nodes[n].x[0] - a.x[0], nodes[n].x[1] - a.x[1], nodes[n].x[2] - a.x[2]);

               Real dotproduct = dot_product(normalize_vector(ab), normalize_vector(an));
               if(dotproduct > 0.9 && dotproduct > bestColinearity) {
                  A = e.corners[c];
                  B = e.corners[(c+1)%3];
                  C = e.corners[(c+2)%3];
                  bestColinearity = dotproduct;
               }
            }


            if(bestColinearity == 0) {
               cerr << "(ionosphere) Stitiching refinement boundaries failed: Element " <<  nodes[n].touchingElements[t] << " does not contain node "
                  << n << " as a corner, yet matching edge not found." << endl;
               continue;
            }

            // We form two elements: AnC and nBC from the old element ABC
            if(A==n || B==n || C==n) {
               cerr << "(ionosphere) ERROR: Trying to split an element at a node that is already it's corner" << endl;
            }

            //Real oldArea = elementArea(nodes[n].touchingElements[t]);
            // Old element modified
            e.corners = {A,n,C};
            //Real newArea1 = elementArea(nodes[n].touchingElements[t]);
            // New element
            Element newElement;
            newElement.corners = {n,B,C};

            uint ne = elements.size();
            elements.push_back(newElement);
            //Real newArea2 = elementArea(ne);

            // Fix touching element lists:
            // Far corner touches both elements
            nodes[C].touchingElements[nodes[C].numTouchingElements++] = ne;
            if(nodes[C].numTouchingElements > MAX_TOUCHING_ELEMENTS) {
               cerr << "(ionosphere) ERROR: node " << C << "'s numTouchingElements (" << nodes[C].numTouchingElements << ") exceeds MAX_TOUCHING_ELEMENTS (= " <<
                        MAX_TOUCHING_ELEMENTS << ")" << endl;
            }

            // Our own node too.
            nodes[n].touchingElements[nodes[n].numTouchingElements++] = ne;
            if(nodes[n].numTouchingElements > MAX_TOUCHING_ELEMENTS) {
               cerr << "(ionosphere) ERROR: node " << n << "'s numTouchingElements [" << nodes[n].numTouchingElements << "] exceeds MAX_TOUCHING_ELEMENTS (= " <<
                        MAX_TOUCHING_ELEMENTS << ")" << endl;
            }

            // One node has been shifted to the other element. Find the old one and change it.
            Node& neighbour=nodes[B];
            for(uint i=0; i<neighbour.numTouchingElements; i++) {
               if(neighbour.touchingElements[i] == nodes[n].touchingElements[t]) {
                  neighbour.touchingElements[i] = ne;
                  continue;
               }

               // Also it's neighbour element nodes might now need their element information updated, if they sit on the B-C line
               Vec3d bc(nodes[C].x[0] - nodes[B].x[0], nodes[C].x[1] - nodes[B].x[1], nodes[C].x[2] - nodes[B].x[2]);
               for(int c=0; c<3; c++) {
                  uint nn=elements[neighbour.touchingElements[i]].corners[c];
                  if(nn == A || nn == B || nn == C || nn==n) {
                     // Skip our own nodes
                     continue;
                  }

                  Vec3d bn(nodes[nn].x[0] - nodes[B].x[0], nodes[nn].x[1] - nodes[B].x[1], nodes[nn].x[2] - nodes[B].x[2]);
                  if(dot_product(normalize_vector(bc), normalize_vector(bn)) > 0.9) {
                     for(uint j=0; j<nodes[nn].numTouchingElements; j++) {
                        if(nodes[nn].touchingElements[j] == nodes[n].touchingElements[t]) {
                           nodes[nn].touchingElements[j] = ne;
                           continue;
                        }
                     }
                  }
               }

               // TODO: What about cases where we refine more than one level at once?
            }
         }
      }
   }

   // Initialize the CG sover by assigning matrix dependency weights
   void SphericalTriGrid::initSolver(bool zeroOut) {

     phiprof::Timer timer {"ionosphere-initSolver"};
     // Zero out parameters
     if(zeroOut) {
        for(uint n=0; n<nodes.size(); n++) {
           for(uint p=ionosphereParameters::SOLUTION; p<ionosphereParameters::N_IONOSPHERE_PARAMETERS-1; p++) {
              Node& N=nodes[n];
              N.parameters[p] = 0;
           }
        }
     } else {
        // Only zero the gradient states
        Real potentialSum=0;
        for(uint n=0; n<nodes.size(); n++) {
           Node& N=nodes[n];
           potentialSum += N.parameters[ionosphereParameters::SOLUTION];
           for(uint p=ionosphereParameters::ZPARAM; p<ionosphereParameters::N_IONOSPHERE_PARAMETERS-1; p++) {
              N.parameters[p] = 0;
           }
        }

        potentialSum /= nodes.size();
        // One option for gauge fixing:
        // Make sure the potential is symmetric around 0 (to prevent it from drifting)
        //for(uint n=0; n<nodes.size(); n++) {
        //   Node& N=nodes[n];
        //   N.parameters[ionosphereParameters::SOLUTION] -= potentialSum;
        //}

     }

     #pragma omp parallel for
     for(uint n=0; n<nodes.size(); n++) {
       addAllMatrixDependencies(n);
     }

     //cerr << "(ionosphere) Solver dependency matrix: " << endl;
     //for(uint n=0; n<nodes.size(); n++) {
     //   for(uint m=0; m<nodes.size(); m++) {

     //      Real val=0;
     //      for(int d=0; d<nodes[n].numDepNodes; d++) {
     //        if(nodes[n].dependingNodes[d] == m) {
     //          val=nodes[n].dependingCoeffs[d];
     //        }
     //      }

     //      cerr << val << "\t";
     //   }
     //   cerr << endl;
     //}

   }

   // Evaluate a nodes' neighbour parameter, averaged through the coupling
   // matrix
   //
   // -> "A times parameter"
   iSolverReal SphericalTriGrid::Atimes(uint nodeIndex, int parameter, bool transpose) {
     iSolverReal retval=0;
     Node& n = nodes[nodeIndex];

     if(transpose) {
        for(uint i=0; i<n.numDepNodes; i++) {
           retval += nodes[n.dependingNodes[i]].parameters[parameter] * n.transposedCoeffs[i];
        }
     } else {
        for(uint i=0; i<n.numDepNodes; i++) {
           retval += nodes[n.dependingNodes[i]].parameters[parameter] * n.dependingCoeffs[i];
        }
     }

     return retval;
   }

   // Evaluate a nodes' own parameter value
   // (If preconditioning is used, this is already adjusted for self-coupling)
   Real SphericalTriGrid::Asolve(uint nodeIndex, int parameter, bool transpose) {

      Node& n = nodes[nodeIndex];

     if(Ionosphere::solverPreconditioning) {
        // Find this nodes' selfcoupling coefficient
        if(transpose) {
           return n.parameters[parameter] / n.transposedCoeffs[0];
        } else {
           return n.parameters[parameter] / n.dependingCoeffs[0];
        }
     } else {
        return n.parameters[parameter];
     }
   }

   // Solve the ionosphere potential using a conjugate gradient solver
   void SphericalTriGrid::solve(
      int &nIterations,
      int &nRestarts,
      Real &residual,
      Real &minPotentialN,
      Real &maxPotentialN,
      Real &minPotentialS,
      Real &maxPotentialS
   ) {

      // Simulations without an ionosphere don't need to bother about this.
      if(nodes.size() == 0) {
         nIterations =0;
         nRestarts = 0;
         residual = 0.;
         minPotentialN = maxPotentialN = minPotentialS = maxPotentialS = 0.;
         return;
      }

      // Ranks that don't participate in ionosphere solving skip this function outright
      if(!isCouplingInwards && !isCouplingOutwards) {
         return;
      }
      initSolver(false);



      nIterations = 0;

      nRestarts = 0;

      for(uint n=0; n<nodes.size(); n++) {

         SphericalTriGrid::Node& N = ionosphereGrid.nodes[n];

         if(fabs(N.x[2]) < Ionosphere::innerRadius * sin(Ionosphere::shieldingLatitude * M_PI / 180.0)) {

            N.parameters[ionosphereParameters::SOLUTION] = 0;

         }

      }


      Eigen::SparseMatrix<Real> potentialSolverMatrix(nodes.size(), nodes.size());

      Eigen::VectorXd vRightHand(nodes.size()), vPhi(nodes.size());

      for(uint n=0; n<nodes.size(); n++) {



         for(uint m=0; m<nodes[n].numDepNodes; m++) { 

            potentialSolverMatrix.insert(n, nodes[n].dependingNodes[m]) = nodes[n].dependingCoeffs[m];

         }



         vRightHand[n] = nodes[n].parameters[ionosphereParameters::SOURCE];

      }

      // gettimeofday(&tStart, NULL);

      potentialSolverMatrix.makeCompressed(); 

      Eigen::BiCGSTAB<Eigen::SparseMatrix<Real> > solver;

      solver.compute(potentialSolverMatrix);

      vPhi = solver.solve(vRightHand); 

      for(uint n=0; n<nodes.size(); n++) { 

         nodes[n].parameters[ionosphereParameters::SOLUTION] = vPhi[n]; 

      }



      nIterations = solver.iterations();

      residual = solver.error();



      for(uint n=0; n<nodes.size(); n++) {

         Node& N=nodes[n];

         if(N.x[2] >= 0) {

            if(N.parameters.at(ionosphereParameters::SOLUTION) < minPotentialN) {

               minPotentialN = N.parameters.at(ionosphereParameters::SOLUTION);

            }

            if(N.parameters.at(ionosphereParameters::SOLUTION) > maxPotentialN) {

               maxPotentialN = N.parameters.at(ionosphereParameters::SOLUTION);

            }

         } else {

            if(N.parameters.at(ionosphereParameters::SOLUTION) < minPotentialS) {

               minPotentialS = N.parameters.at(ionosphereParameters::SOLUTION);

            }

            if(N.parameters.at(ionosphereParameters::SOLUTION) > maxPotentialS) {

               maxPotentialS = N.parameters.at(ionosphereParameters::SOLUTION);

            }

         }

      }

     }

   void SphericalTriGrid::solveInternal(
      int & iteration,
      int & nRestarts,
      Real & minerr,
      Real & minPotentialN,
      Real & maxPotentialN,
      Real & minPotentialS,
      Real & maxPotentialS
   ) {
      std::vector<iSolverReal> effectiveSource(nodes.size());

      // for loop reduction variables, declared before omp parallel region
      iSolverReal akden;
      iSolverReal bknum;
      iSolverReal potentialInt;
      iSolverReal sourcenorm;
      iSolverReal residualnorm;
      minPotentialN = minPotentialS = std::numeric_limits<iSolverReal>::max();
      maxPotentialN = maxPotentialS = std::numeric_limits<iSolverReal>::lowest();
#ifdef IONOSPHERE_SORTED_SUMS
      std::multiset<iSolverReal> set_neg, set_pos;
#endif

#ifdef IONOSPHERE_SORTED_SUMS
#pragma omp parallel shared(akden,bknum,potentialInt,sourcenorm,residualnorm,effectiveSource,minPotentialN,maxPotentialN,minPotentialS,maxPotentialS,set_neg,set_pos)
#else
#pragma omp parallel shared(akden,bknum,potentialInt,sourcenorm,residualnorm,effectiveSource,minPotentialN,maxPotentialN,minPotentialS,maxPotentialS)
#endif
{

      // thread variables, initialised here
      iSolverReal err = 0;
      iSolverReal thread_minerr = std::numeric_limits<iSolverReal>::max();
      int thread_iteration = iteration;
      int thread_nRestarts = nRestarts;
#ifdef IONOSPHERE_SORTED_SUMS
      std::multiset<iSolverReal> thread_set_neg, thread_set_pos;
#endif

      iSolverReal bkden = 1;
      int failcount=0;
      int counter = 0;

      #pragma omp single
      {
         sourcenorm = 0;
      }
      // Calculate sourcenorm and initial residual estimate
#ifdef IONOSPHERE_SORTED_SUMS
      #pragma omp for
#else
      #pragma omp for reduction(+:sourcenorm)
#endif
      for(uint n=0; n<nodes.size(); n++) {
         Node& N=nodes[n];
         // Set gauge-pinned nodes to their fixed potential
         //if(ionosphereGrid.gaugeFixing == Pole && n == 0) {
         //   effectiveSource[n] = 0;
         //} else if(ionosphereGrid.gaugeFixing == Equator && fabs(N.x[2]) < Ionosphere::innerRadius * sin(Ionosphere::shieldingLatitude * M_PI / 180.0)) {
         //   effectiveSource[n] = 0;
         //}  else {
            iSolverReal source = N.parameters[ionosphereParameters::SOURCE];
            effectiveSource[n] = source;
         //}
         if(source != 0) {
#ifdef IONOSPHERE_SORTED_SUMS
            thread_set_pos.insert(source*source);
#else
            sourcenorm += source*source;
#endif
         }
         N.parameters.at(ionosphereParameters::RESIDUAL) = source - Atimes(n, ionosphereParameters::SOLUTION);
         N.parameters.at(ionosphereParameters::BEST_SOLUTION) = N.parameters.at(ionosphereParameters::SOLUTION);
         if(Ionosphere::solverUseMinimumResidualVariant) {
            N.parameters.at(ionosphereParameters::RRESIDUAL) = Atimes(n, ionosphereParameters::RESIDUAL, false);
         } else {
            N.parameters.at(ionosphereParameters::RRESIDUAL) = N.parameters.at(ionosphereParameters::RESIDUAL);
         }
      }
#ifdef IONOSPHERE_SORTED_SUMS
      #pragma omp critical
      {
         set_pos.insert(thread_set_pos.begin(), thread_set_pos.end());
      }
#endif
      #pragma omp barrier
      #pragma omp single
      {
#ifdef IONOSPHERE_SORTED_SUMS
         for(auto it = set_pos.crbegin(); it != set_pos.crend(); it++) {
            sourcenorm += *it;
         }
#endif
         sourcenorm = sqrt(sourcenorm);
      }
      bool skipSolve = false;
      // Abort if there is nothing to solve.
      if(sourcenorm == 0) {
         skipSolve = true;
      }

      #pragma omp for
      for(uint n=0; n<nodes.size(); n++) {
         Node& N=nodes[n];
         N.parameters.at(ionosphereParameters::ZPARAM) = Asolve(n,ionosphereParameters::RESIDUAL, false);
      }

      while(!skipSolve && thread_iteration < Ionosphere::solverMaxIterations) {
         thread_iteration++;
         counter++;

         #pragma omp for
         for(uint n=0; n<nodes.size(); n++) {
            Node& N=nodes[n];
            N.parameters[ionosphereParameters::ZZPARAM] = Asolve(n,ionosphereParameters::RRESIDUAL, true);
         }

         // Calculate bk and gradient vector p
         #pragma omp single
         {
            bknum = 0;
#ifdef IONOSPHERE_SORTED_SUMS
            set_pos.clear();
            set_neg.clear();
#endif
         }
#ifdef IONOSPHERE_SORTED_SUMS
         thread_set_pos.clear();
         thread_set_neg.clear();
         #pragma omp for
#else
         #pragma omp for reduction(+:bknum)
#endif
         for(uint n=0; n<nodes.size(); n++) {
            Node& N=nodes[n];
            const iSolverReal incr = N.openFieldLine * N.parameters[ionosphereParameters::RRESIDUAL];
#ifdef IONOSPHERE_SORTED_SUMS
            if(incr < 0) {
               thread_set_neg.insert(incr);
            }
            if(incr > 0) {
               thread_set_pos.insert(incr);
            }
#else
            bknum += incr;
#endif
         }

#ifdef IONOSPHERE_SORTED_SUMS
         #pragma omp critical
         {
            set_neg.insert(thread_set_neg.begin(), thread_set_neg.end());
            set_pos.insert(thread_set_pos.begin(), thread_set_pos.end());
         }
         #pragma omp barrier
         #pragma omp single
         {
            iSolverReal bknum_pos = 0;
            iSolverReal bknum_neg = 0;
            for(auto it = set_neg.cbegin(); it != set_neg.cend(); it++) {
               bknum_neg += *it;
            }
            for(auto it = set_pos.crbegin(); it != set_pos.crend(); it++) {
               bknum_pos += *it;
            }
            bknum = bknum_neg + bknum_pos;
         }
#endif

         if(counter == 1) {
            // Just use the gradient vector as-is, starting from the best known solution
            #pragma omp for
            for(uint n=0; n<nodes.size(); n++) {
               Node& N=nodes[n];
               N.parameters[ionosphereParameters::PPARAM] = N.openFieldLine;
               N.parameters[ionosphereParameters::PPPARAM] = N.parameters[ionosphereParameters::ZZPARAM];
            }
         } else {
            // Perform gram-smith orthogonalization to get conjugate gradient
            iSolverReal bk = bknum / bkden;
            #pragma omp for
            for(uint n=0; n<nodes.size(); n++) {
               Node& N=nodes[n];
               N.parameters[ionosphereParameters::PPARAM] *= bk;
               N.parameters[ionosphereParameters::PPARAM] += N.openFieldLine;
               N.parameters[ionosphereParameters::PPPARAM] *= bk;
               N.parameters[ionosphereParameters::PPPARAM] += N.parameters[ionosphereParameters::ZZPARAM];
            }
         }
         bkden = bknum;
         if(bkden == 0) {
            bkden = 1;
         }


         // Calculate ak, new solution and new residual
         #pragma omp single
         {
            akden = 0;
#ifdef IONOSPHERE_SORTED_SUMS
            set_neg.clear();
            set_pos.clear();
#endif
         }
#ifdef IONOSPHERE_SORTED_SUMS
         thread_set_neg.clear();
         thread_set_pos.clear();
         #pragma omp for
#else
         #pragma omp for reduction(+:akden)
#endif
         for(uint n=0; n<nodes.size(); n++) {
            Node& N=nodes[n];
            iSolverReal zparam = Atimes(n, ionosphereParameters::PPARAM, false);
            N.openFieldLine = zparam;
            iSolverReal incr = zparam * N.parameters[ionosphereParameters::PPPARAM];
#ifdef IONOSPHERE_SORTED_SUMS
            if(incr < 0) {
               thread_set_neg.insert(incr);
            }
            if(incr > 0) {
               thread_set_pos.insert(incr);
            }
#else
            akden += incr;
#endif
            N.parameters[ionosphereParameters::ZZPARAM] = Atimes(n,ionosphereParameters::PPPARAM, true);
         }
#ifdef IONOSPHERE_SORTED_SUMS
         #pragma omp critical
         {
            set_neg.insert(thread_set_neg.begin(), thread_set_neg.end());
            set_pos.insert(thread_set_pos.begin(), thread_set_pos.end());
         }
         #pragma omp barrier
         #pragma omp single
         {
            iSolverReal akden_pos = 0;
            iSolverReal akden_neg = 0;
            for(auto it = set_neg.cbegin(); it != set_neg.cend(); it++) {
               akden_neg += *it;
            }
            for(auto it = set_pos.crbegin(); it != set_pos.crend(); it++) {
               akden_pos += *it;
            }
            akden = akden_neg + akden_pos;
         }
#endif
         iSolverReal ak=bknum/akden;

         #pragma omp for
         for(uint n=0; n<nodes.size(); n++) {
            Node& N=nodes[n];
            N.parameters[ionosphereParameters::SOLUTION] += ak * N.parameters[ionosphereParameters::PPARAM];
            if(ionosphereGrid.gaugeFixing == Pole && n == 0) {
               N.parameters[ionosphereParameters::SOLUTION] = 0;
            } else if(ionosphereGrid.gaugeFixing == Equator && fabs(N.x[2]) < Ionosphere::innerRadius * sin(Ionosphere::shieldingLatitude * M_PI / 180.0)) {
               N.parameters[ionosphereParameters::SOLUTION] = 0;
            }
         }

         // Rebalance the potential by calculating its area integral
         if(ionosphereGrid.gaugeFixing == Integral) {
            #pragma omp single
            {
               potentialInt = 0;
            }
            #pragma omp for reduction(+:potentialInt)
            for(uint e=0; e<elements.size(); e++) {
               Real area = elementArea(e);
               Real effPotential = 0;
               for(int c=0; c<3; c++) {
                  effPotential += nodes[elements[e].corners[c]].parameters[ionosphereParameters::SOLUTION];
               }

               potentialInt += effPotential * area;
            }
            // Calculate average potential on the sphere
            #pragma omp single
            {
               potentialInt /= 4. * M_PI * Ionosphere::innerRadius * Ionosphere::innerRadius;
            }

            // Offset potentials to make it zero
            #pragma omp for
            for(uint n=0; n<nodes.size(); n++) {
               Node& N=nodes[n];
               N.parameters[ionosphereParameters::SOLUTION] -= potentialInt;
            }
         }

         #pragma omp single
         {
            residualnorm = 0;
#ifdef IONOSPHERE_SORTED_SUMS
            set_pos.clear();
#endif
         }
#ifdef IONOSPHERE_SORTED_SUMS
         thread_set_pos.clear();
         #pragma omp for
#else
         #pragma omp for reduction(+:residualnorm)
#endif
         for(uint n=0; n<nodes.size(); n++) {
            Node& N=nodes[n];
            // Calculate residual of the new solution. The faster way to do this would be
            //
            // iSolverReal newresid = N.parameters[ionosphereParameters::RESIDUAL] - ak * N.openFieldLine;
            // and
            // N.parameters[ionosphereParameters::RRESIDUAL] -= ak * N.parameters[ionosphereParameters::ZZPARAM];
            //
            // but doing so leads to numerical inaccuracy due to roundoff errors
            // when iteration counts are high (because, for example, mesh node count is high and the matrix condition is bad).
            // See https://en.wikipedia.org/wiki/Conjugate_gradient_method#Explicit_residual_calculation
            iSolverReal newresid = effectiveSource[n] - Atimes(n, ionosphereParameters::SOLUTION);
            if( (ionosphereGrid.gaugeFixing == Pole && n == 0) || (ionosphereGrid.gaugeFixing == Equator && fabs(N.x[2]) < Ionosphere::innerRadius * sin(Ionosphere::shieldingLatitude * M_PI / 180.0))) {
               // Don't calculate residual for gauge-pinned nodes
               N.parameters[ionosphereParameters::RESIDUAL] = 0;
               N.parameters[ionosphereParameters::RRESIDUAL] = 0;
            } else {
               N.parameters[ionosphereParameters::RESIDUAL] = newresid;
               N.parameters[ionosphereParameters::RRESIDUAL] = effectiveSource[n] - Atimes(n, ionosphereParameters::SOLUTION, true);
#ifdef IONOSPHERE_SORTED_SUMS
               thread_set_pos.insert(newresid*newresid);
#else
               residualnorm += newresid*newresid;
#endif
            }
         }

#ifdef IONOSPHERE_SORTED_SUMS
         #pragma omp critical
         {
            set_pos.insert(thread_set_pos.begin(), thread_set_pos.end());
         }
         #pragma omp barrier
         #pragma omp single
         {
            for(auto it = set_pos.crbegin(); it != set_pos.crend(); it++) {
               residualnorm += *it;
            }
         }
#endif

         #pragma omp for
         for(uint n=0; n<nodes.size(); n++) {
            Node& N=nodes[n];
            N.openFieldLine = Asolve(n, ionosphereParameters::RESIDUAL, false);
         }

         // See if this solved the potential better than before
         err = sqrt(residualnorm)/sourcenorm;


         if(err < thread_minerr) {
            // If yes, this is our new best solution
            #pragma omp for
            for(uint n=0; n<nodes.size(); n++) {
               Node& N=nodes[n];
               N.parameters[ionosphereParameters::BEST_SOLUTION] = N.parameters[ionosphereParameters::SOLUTION];
            }
            thread_minerr = err;
            failcount = 0;
         } else {
            // If no, keep going with the best one
            #pragma omp for
            for(uint n=0; n<nodes.size(); n++) {
               Node& N=nodes[n];
               N.parameters[ionosphereParameters::SOLUTION] = N.parameters[ionosphereParameters::BEST_SOLUTION];
            }
            failcount++;
         }

         if(thread_minerr < Ionosphere::solverRelativeL2ConvergenceThreshold) {
            break;
         }
         if(failcount > Ionosphere::solverMaxFailureCount || err > Ionosphere::solverMaxErrorGrowthFactor*thread_minerr) {
            thread_nRestarts++;
            break;
         }
      } // while

      int threadID = 0;
#ifdef _OPENMP
      threadID = omp_get_thread_num();
#endif
      if(skipSolve && threadID == 0) {
         // sourcenorm was zero, we return zero; return is not allowed inside threaded region
         minerr = 0;
         minPotentialN = 0;
         maxPotentialN = 0;
         minPotentialS = 0;
         maxPotentialS = 0;
      } else {
         #pragma omp for reduction(max:maxPotentialN,maxPotentialS) reduction(min:minPotentialN,minPotentialS)
         for(uint n=0; n<nodes.size(); n++) {
            Node& N=nodes.at(n);
            N.parameters.at(ionosphereParameters::SOLUTION) = N.parameters.at(ionosphereParameters::BEST_SOLUTION);
            if(N.x.at(2) > 0) {
               minPotentialN = min(minPotentialN, N.parameters.at(ionosphereParameters::SOLUTION));
               maxPotentialN = max(maxPotentialN, N.parameters.at(ionosphereParameters::SOLUTION));
            } else {
               minPotentialS = min(minPotentialS, N.parameters.at(ionosphereParameters::SOLUTION));
               maxPotentialS = max(maxPotentialS, N.parameters.at(ionosphereParameters::SOLUTION));
            }
         }
         // Get out the ones we need before exiting the parallel region
         if(threadID == 0) {
            minerr = thread_minerr;
            iteration = thread_iteration;
            nRestarts = thread_nRestarts;
         }
      }

} // #pragma omp parallel

   }

   // Actual ionosphere object implementation

   Ionosphere::Ionosphere(): SysBoundaryCondition() { }

   Ionosphere::~Ionosphere() { }

   void Ionosphere::addParameters() {
      Readparameters::add("ionosphere.centerX", "X coordinate of ionosphere center (m)", 0.0);
      Readparameters::add("ionosphere.centerY", "Y coordinate of ionosphere center (m)", 0.0);
      Readparameters::add("ionosphere.centerZ", "Z coordinate of ionosphere center (m)", 0.0);
      Readparameters::add("ionosphere.radius", "Radius of the inner simulation boundary (unit is assumed to be R_E if value < 1000, otherwise m).", 1.0e7);
      Readparameters::add("ionosphere.innerRadius", "Radius of the ionosphere model (m).", physicalconstants::R_E + 100e3);
      Readparameters::add("ionosphere.geometry", "Select the geometry of the ionosphere, 0: inf-norm (diamond), 1: 1-norm (square), 2: 2-norm (circle, DEFAULT), 3: 2-norm cylinder aligned with y-axis, use with polar plane/line dipole.", 2);
      Readparameters::add("ionosphere.precedence", "Precedence value of the ionosphere system boundary condition (integer), the higher the stronger.", 2);
      Readparameters::add("ionosphere.reapplyUponRestart", "If 0 (default), keep going with the state existing in the restart file. If 1, calls again applyInitialState. Can be used to change boundary condition behaviour during a run.", 0);
      Readparameters::add("ionosphere.baseShape", "Select the seed mesh geometry for the spherical ionosphere grid. Options are: fromFile, sphericalFibonacci, tetrahedron, icosahedron.",std::string("sphericalFibonacci"));
      Readparameters::add("ionosphere.conductivityModel", "Select ionosphere conductivity tensor construction model. Options are: 0=GUMICS style (Vertical B, only SigmaH and SigmaP), 1=Ridley et al 2004 (1000 mho longitudinal conductivity), 2=Koskinen 2011 full conductivity tensor.", 0);
      Readparameters::add("ionosphere.ridleyParallelConductivity", "Constant parallel conductivity value. 1000 mho is given without justification by Ridley et al 2004.", 1000);
      Readparameters::add("ionosphere.fibonacciNodeNum", "Number of nodes in the spherical fibonacci mesh.",256);
      Readparameters::add("ionosphere.gridFilePath", "Path to the ionosphere grid mesh OBJ or VTK legacy file, if loading grid from file.",std::string(""));
      Readparameters::addComposing("ionosphere.refineMinLatitude", "Refine the grid polewards of the given latitude. Multiple of these lines can be given for successive refinement, paired up with refineMaxLatitude lines.");
      Readparameters::addComposing("ionosphere.refineMaxLatitude", "Refine the grid equatorwards of the given latitude. Multiple of these lines can be given for successive refinement, paired up with refineMinLatitude lines.");
      Readparameters::add("ionosphere.atmosphericModelFile", "Filename to read the MSIS atmosphere data from (default: NRLMSIS.dat)", std::string("NRLMSIS.dat"));
      Readparameters::add("ionosphere.recombAlpha", "Ionospheric recombination parameter (m^3/s)", 2.4e-13); // Default value from Schunck & Nagy, Table 8.5
      Readparameters::add("ionosphere.ionizationModel", "Ionospheric electron production rate model. Options are: Rees1963, Rees1989, SergienkoIvanov (default), Robinson2020.", std::string("SergienkoIvanov"));
      Readparameters::add("ionosphere.innerBoundaryVDFmode", "Inner boundary VDF construction method. Options ar: FixedMoments, AverageMoments, AverageAllMoments, CopyAndLosscone.", std::string("FixedMoments"));
      Readparameters::add("ionosphere.F10_7", "Solar 10.7 cm radio flux (sfu = 10^{-22} W/m^2)", 100);
      Readparameters::add("ionosphere.backgroundIonisation", "Background ionoisation due to cosmic rays (mho)", 0.5);
      Readparameters::add("ionosphere.solverMaxIterations", "Maximum number of iterations for the conjugate gradient solver", 2000);
      Readparameters::add("ionosphere.solverRelativeL2ConvergenceThreshold", "Convergence threshold for the relative L2 metric", 1e-6);
      Readparameters::add("ionosphere.solverMaxFailureCount", "Maximum number of iterations allowed to diverge before restarting the ionosphere solver", 5);
      Readparameters::add("ionosphere.solverMaxErrorGrowthFactor", "Maximum allowed factor of growth with respect to the minimum error before restarting the ionosphere solver", 100);
      Readparameters::add("ionosphere.solverGaugeFixing", "Gauge fixing method of the ionosphere solver. Options are: pole, integral, equator", std::string("equator"));
      Readparameters::add("ionosphere.shieldingLatitude", "Latitude below which the potential is set to zero in the equator gauge fixing scheme (degree)", 70);
      Readparameters::add("ionosphere.solverPreconditioning", "Use preconditioning for the solver? (0/1)", 1);
      Readparameters::add("ionosphere.solverUseMinimumResidualVariant", "Use minimum residual variant", 0);
      Readparameters::add("ionosphere.solverToggleMinimumResidualVariant", "Toggle use of minimum residual variant at every solver restart", 0);
      Readparameters::add("ionosphere.earthAngularVelocity", "Angular velocity of inner boundary convection, in rad/s", 7.2921159e-5);
      Readparameters::add("ionosphere.plasmapauseL", "L-shell at which the plasmapause resides (for corotation)", 5.);
      Readparameters::add("ionosphere.downmapRadius", "Radius from which FACs are coupled down into the ionosphere. Units are assumed to be RE if value < 1000, otherwise m. If -1: use inner boundary cells.", -1.);
      Readparameters::add("ionosphere.unmappedNodeRho", "Electron density of ionosphere nodes that do not connect to the magnetosphere domain.", 1e4);
      Readparameters::add("ionosphere.unmappedNodeTe", "Electron temperature of ionosphere nodes that do not connect to the magnetosphere domain.", 1e6);
      Readparameters::add("ionosphere.couplingTimescale", "Magnetosphere->Ionosphere coupling timescale (seconds, 0=immediate coupling", 1.);
      Readparameters::add("ionosphere.couplingInterval", "Time interval at which the ionosphere is solved (seconds)", 0);

      // Per-population parameters
      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
         const std::string& pop = getObjectWrapper().particleSpecies[i].name;
         Readparameters::add(pop + "_ionosphere.rho", "Number density of the ionosphere (m^-3)", 0.0);
         Readparameters::add(pop + "_ionosphere.T", "Temperature of the ionosphere (K)", 0.0);
         Readparameters::add(pop + "_ionosphere.VX0", "Bulk velocity of ionospheric distribution function in X direction (m/s)", 0.0);
         Readparameters::add(pop + "_ionosphere.VY0", "Bulk velocity of ionospheric distribution function in X direction (m/s)", 0.0);
         Readparameters::add(pop + "_ionosphere.VZ0", "Bulk velocity of ionospheric distribution function in X direction (m/s)", 0.0);
      }
   }

   void Ionosphere::getParameters() {

      Readparameters::get("ionosphere.centerX", this->center[0]);
      Readparameters::get("ionosphere.centerY", this->center[1]);
      Readparameters::get("ionosphere.centerZ", this->center[2]);
      Readparameters::get("ionosphere.radius", this->radius);
      if(radius < 1000.) {
         // If radii are < 1000, assume they are given in R_E.
         radius *= physicalconstants::R_E;
      }

      Readparameters::get("ionosphere.geometry", this->geometry);
      Readparameters::get("ionosphere.precedence", this->precedence);

      uint reapply;
      Readparameters::get("ionosphere.reapplyUponRestart", reapply);
      this->applyUponRestart = (reapply == 1);

      Readparameters::get("ionosphere.baseShape",baseShape);

      int cm;
      Readparameters::get("ionosphere.conductivityModel", cm);
      conductivityModel = static_cast<Ionosphere::IonosphereConductivityModel>(cm);

      std::string VDFmodeString;
      Readparameters::get("ionosphere.innerBoundaryVDFmode", VDFmodeString);
      if(VDFmodeString == "FixedMoments") {
         boundaryVDFmode = FixedMoments;
      } else if(VDFmodeString == "AverageMoments") {
         boundaryVDFmode = AverageMoments;
      } else if(VDFmodeString == "AverageAllMoments") {
         boundaryVDFmode = AverageAllMoments;
      } else if(VDFmodeString == "CopyAndLosscone") {
         boundaryVDFmode = CopyAndLosscone;
      } else {
         cerr << "(IONOSPHERE) Unknown inner boundary VDF mode \"" << VDFmodeString << "\". Aborting." << endl;
         abort();
      }
      Readparameters::get("ionosphere.ridleyParallelConductivity", ridleyParallelConductivity);
      Readparameters::get("ionosphere.fibonacciNodeNum",fibonacciNodeNum);
      Readparameters::get("ionosphere.gridFilePath",path);
      Readparameters::get("ionosphere.solverMaxIterations", solverMaxIterations);
      Readparameters::get("ionosphere.solverRelativeL2ConvergenceThreshold", solverRelativeL2ConvergenceThreshold);
      Readparameters::get("ionosphere.solverMaxFailureCount", solverMaxFailureCount);
      Readparameters::get("ionosphere.solverMaxErrorGrowthFactor", solverMaxErrorGrowthFactor);
      std::string gaugeFixingString;
      Readparameters::get("ionosphere.solverGaugeFixing", gaugeFixingString);
      if(gaugeFixingString == "pole") {
         ionosphereGrid.gaugeFixing = SphericalTriGrid::Pole;
      } else if (gaugeFixingString == "integral") {
         ionosphereGrid.gaugeFixing = SphericalTriGrid::Integral;
      } else if (gaugeFixingString == "equator") {
         ionosphereGrid.gaugeFixing = SphericalTriGrid::Equator;
      } else if (gaugeFixingString == "None") {
         ionosphereGrid.gaugeFixing = SphericalTriGrid::None;
      } else {
         cerr << "(IONOSPHERE) Unknown solver gauge fixing method \"" << gaugeFixingString << "\". Aborting." << endl;
         abort();
      }
      Readparameters::get("ionosphere.shieldingLatitude", shieldingLatitude);
      Readparameters::get("ionosphere.solverPreconditioning", solverPreconditioning);
      Readparameters::get("ionosphere.solverUseMinimumResidualVariant", solverUseMinimumResidualVariant);
      Readparameters::get("ionosphere.solverToggleMinimumResidualVariant", solverToggleMinimumResidualVariant);
      Readparameters::get("ionosphere.earthAngularVelocity", earthAngularVelocity);
      Readparameters::get("ionosphere.plasmapauseL", plasmapauseL);
      Readparameters::get("ionosphere.couplingTimescale",couplingTimescale);
      Readparameters::get("ionosphere.couplingInterval", couplingInterval);
      Readparameters::get("ionosphere.downmapRadius",downmapRadius);
      if(downmapRadius < 1000.) {
         downmapRadius *= physicalconstants::R_E;
      }
      if(downmapRadius < radius) {
         downmapRadius = radius;
      }
      Readparameters::get("ionosphere.unmappedNodeRho", unmappedNodeRho);
      Readparameters::get("ionosphere.unmappedNodeTe",  unmappedNodeTe);
      Readparameters::get("ionosphere.innerRadius", innerRadius);
      FieldTracing::fieldTracingParameters.innerBoundaryRadius = this->innerRadius;
      Readparameters::get("ionosphere.refineMinLatitude",refineMinLatitudes);
      Readparameters::get("ionosphere.refineMaxLatitude",refineMaxLatitudes);
      Readparameters::get("ionosphere.atmosphericModelFile",atmosphericModelFile);
      Readparameters::get("ionosphere.recombAlpha",recombAlpha);
      std::string ionizationModelString;
      Readparameters::get("ionosphere.ionizationModel", ionizationModelString);
      if(ionizationModelString == "Rees1963") {
         ionosphereGrid.ionizationModel = SphericalTriGrid::Rees1963;
      } else if(ionizationModelString == "Rees1989") {
         ionosphereGrid.ionizationModel = SphericalTriGrid::Rees1989;
      } else if(ionizationModelString == "SergienkoIvanov") {
         ionosphereGrid.ionizationModel = SphericalTriGrid::SergienkoIvanov;
      } else if (ionizationModelString == "Robinson2020") {
         ionosphereGrid.ionizationModel = SphericalTriGrid::Robinson2020;
      } else if (ionizationModelString == "Juusola2025") {
         ionosphereGrid.ionizationModel = SphericalTriGrid::Juusola2025;
      } else {
         cerr << "(IONOSPHERE) Unknown ionization production model \"" << ionizationModelString << "\". Aborting." << endl;
         abort();
      }
      Readparameters::get("ionosphere.F10_7",F10_7);
      Readparameters::get("ionosphere.backgroundIonisation",backgroundIonisation);

      for(uint i=0; i< getObjectWrapper().particleSpecies.size(); i++) {
        const std::string& pop = getObjectWrapper().particleSpecies[i].name;
        IonosphereSpeciesParameters sP;

        Readparameters::get(pop + "_ionosphere.rho", sP.rho);
        Readparameters::get(pop + "_ionosphere.VX0", sP.V0[0]);
        Readparameters::get(pop + "_ionosphere.VY0", sP.V0[1]);
        Readparameters::get(pop + "_ionosphere.VZ0", sP.V0[2]);
        Readparameters::get(pop + "_ionosphere.T", sP.T);

        // Failsafe, if density or temperature is zero, read from Magnetosphere
        // (compare the corresponding verbose handling in projects/Magnetosphere/Magnetosphere.cpp)
        if(sP.T == 0) {
           Readparameters::get(pop + "_Magnetosphere.T", sP.T);
        }
        if(sP.rho == 0) {
           Readparameters::get(pop + "_Magnetosphere.rho", sP.rho);
        }

        speciesParams.push_back(sP);
      }
   }

   void Ionosphere::initSysBoundary(
      creal& t,
      Project &project
   ) {
      getParameters();
      dynamic = false;

      // Sanity check: the ionosphere only makes sense in 3D simulations
      if(P::xcells_ini == 1 || P::ycells_ini == 1 || P::zcells_ini == 1) {
         cerr << "*************************************************" << endl;
         cerr << "* BIG FAT IONOSPHERE ERROR:                     *" << endl;
         cerr << "*                                               *" << endl;
         cerr << "* You are trying to run a 2D simulation with an *" << endl;
         cerr << "* ionosphere inner boundary. This won't work.   *" << endl;
         cerr << "*                                               *" << endl;
         cerr << "* Most likely, your config file needs to be up- *" << endl;
         cerr << "* dated, changing all mentions of \"ionosphere\"  *" << endl;
         cerr << "* to \"copysphere\".                        *" << endl;
         cerr << "*                                               *" << endl;
         cerr << "* This simulation will now crash in the friend- *" << endl;
         cerr << "* liest way possible.                           *" << endl;
         cerr << "*************************************************" << endl;
         abort();
      }

      // Initialize ionosphere mesh base shape
      if(baseShape == "icosahedron") {
         ionosphereGrid.initializeIcosahedron();
      } else if(baseShape == "tetrahedron") {
         ionosphereGrid.initializeTetrahedron();
      } else if(baseShape == "sphericalFibonacci") {
         ionosphereGrid.initializeSphericalFibonacci(fibonacciNodeNum);
      } else if(baseShape == "fromFile") {
         ionosphereGrid.initializeGridFromFile(path);
      } else {
         cerr << "(IONOSPHERE) Unknown mesh base shape \"" << baseShape << "\". Aborting." << endl;
         abort();
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

      // Refine the mesh between the given latitudes
      for(uint i=0; i< max(refineMinLatitudes.size(), refineMaxLatitudes.size()); i++) {
         Real lmin;
         if(i < refineMinLatitudes.size()) {
            lmin = refineMinLatitudes[i];
         } else {
            lmin = 0.;
         }
         Real lmax;
         if(i < refineMaxLatitudes.size()) {
            lmax = refineMaxLatitudes[i];
         } else {
            lmax = 90.;
         }
         refineBetweenLatitudes(lmin, lmax);
      }
      ionosphereGrid.stitchRefinementInterfaces();

      // Set up ionospheric atmosphere model
      ionosphereGrid.readAtmosphericModelFile(atmosphericModelFile.c_str());

      // iniSysBoundary is only called once, generateTemplateCell must
      // init all particle species
      generateTemplateCell(project);
   }

   static Real getR(creal x,creal y,creal z, uint geometry, Real center[3]) {

      Real r;

      switch(geometry) {
      case 0:
         // infinity-norm, result is a diamond/square with diagonals aligned on the axes in 2D
         r = fabs(x-center[0]) + fabs(y-center[1]) + fabs(z-center[2]);
         break;
      case 1:
         // 1-norm, result is is a grid-aligned square in 2D
         r = max(max(fabs(x-center[0]), fabs(y-center[1])), fabs(z-center[2]));
         break;
      case 2:
         // 2-norm (Cartesian), result is a circle in 2D
         r = sqrt((x-center[0])*(x-center[0]) + (y-center[1])*(y-center[1]) + (z-center[2])*(z-center[2]));
         break;
      case 3:
         // 2-norm (Cartesian) cylinder aligned on y-axis
         r = sqrt((x-center[0])*(x-center[0]) + (z-center[2])*(z-center[2]));
         break;
      default:
         std::cerr << __FILE__ << ":" << __LINE__ << ":" << "ionosphere.geometry has to be 0, 1 or 2." << std::endl;
         abort();
      }

      return r;
   }

   void Ionosphere::assignSysBoundary(dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
                                      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid) {
      const vector<CellID>& cells = getLocalCells();
      for(uint i=0; i<cells.size(); i++) {
         if(mpiGrid[cells[i]]->sysBoundaryFlag == sysboundarytype::DO_NOT_COMPUTE) {
            continue;
         }

         creal* const cellParams = &(mpiGrid[cells[i]]->parameters[0]);
         creal dx = cellParams[CellParams::DX];
         creal dy = cellParams[CellParams::DY];
         creal dz = cellParams[CellParams::DZ];
         creal x = cellParams[CellParams::XCRD] + 0.5*dx;
         creal y = cellParams[CellParams::YCRD] + 0.5*dy;
         creal z = cellParams[CellParams::ZCRD] + 0.5*dz;

         if(getR(x,y,z,this->geometry,this->center) < this->radius) {
            mpiGrid[cells[i]]->sysBoundaryFlag = this->getIndex();
         }
      }

   }

   void Ionosphere::applyInitialState(
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & perBGrid,
      FsGrid<std::array<Real, fsgrids::bgbfield::N_BGB>, FS_STENCIL_WIDTH>& BgBGrid,
      Project &project
   ) {
      const vector<CellID>& cells = getLocalCells();
      //#pragma omp parallel for
      for (uint i=0; i<cells.size(); ++i) {
         SpatialCell* cell = mpiGrid[cells[i]];
         if (cell->sysBoundaryFlag != this->getIndex()) continue;

         for (uint popID=0; popID<getObjectWrapper().particleSpecies.size(); ++popID) {
            setCellFromTemplate(cell,popID);
            #ifdef DEBUG_VLASIATOR
            // Verify current mesh and blocks
            if (!cell->checkMesh(popID)) {
               printf("ERROR in vmesh check: %s at %d\n",__FILE__,__LINE__);
            }
            #endif
         }
      }
   }

   std::array<Real, 3> Ionosphere::fieldSolverGetNormalDirection(
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      cint i,
      cint j,
      cint k
   ) {
      phiprof::Timer timer {"Ionosphere::fieldSolverGetNormalDirection"};
      std::array<Real, 3> normalDirection{{ 0.0, 0.0, 0.0 }};

      static creal DIAG2 = 1.0 / sqrt(2.0);
      static creal DIAG3 = 1.0 / sqrt(3.0);

      creal dx = technicalGrid.DX;
      creal dy = technicalGrid.DY;
      creal dz = technicalGrid.DZ;
      const std::array<FsGridTools::FsIndex_t, 3> globalIndices = technicalGrid.getGlobalIndices(i,j,k);
      creal x = P::xmin + (convert<Real>(globalIndices[0])+0.5)*dx;
      creal y = P::ymin + (convert<Real>(globalIndices[1])+0.5)*dy;
      creal z = P::zmin + (convert<Real>(globalIndices[2])+0.5)*dz;
      creal xsign = divideIfNonZero(x, fabs(x));
      creal ysign = divideIfNonZero(y, fabs(y));
      creal zsign = divideIfNonZero(z, fabs(z));

      Real length = 0.0;

      if (Parameters::xcells_ini == 1) {
         if (Parameters::ycells_ini == 1) {
            if (Parameters::zcells_ini == 1) {
               // X,Y,Z
               std::cerr << __FILE__ << ":" << __LINE__ << ":" << "What do you expect to do with a single-cell simulation of ionosphere boundary type? Stop kidding." << std::endl;
               abort();
               // end of X,Y,Z
            } else {
               // X,Y
               normalDirection[2] = zsign;
               // end of X,Y
            }
         } else if (Parameters::zcells_ini == 1) {
            // X,Z
            normalDirection[1] = ysign;
            // end of X,Z
         } else {
            // X
            switch(this->geometry) {
               case 0:
                  normalDirection[1] = DIAG2*ysign;
                  normalDirection[2] = DIAG2*zsign;
                  break;
               case 1:
                  if(fabs(y) == fabs(z)) {
                     normalDirection[1] = ysign*DIAG2;
                     normalDirection[2] = zsign*DIAG2;
                     break;
                  }
                  if(fabs(y) > (this->radius - dy)) {
                     normalDirection[1] = ysign;
                     break;
                  }
                  if(fabs(z) > (this->radius - dz)) {
                     normalDirection[2] = zsign;
                     break;
                  }
                  if(fabs(y) > (this->radius - 2.0*dy)) {
                     normalDirection[1] = ysign;
                     break;
                  }
                  if(fabs(z) > (this->radius - 2.0*dz)) {
                     normalDirection[2] = zsign;
                     break;
                  }
                  break;
               case 2:
                  length = sqrt(y*y + z*z);
                  normalDirection[1] = y / length;
                  normalDirection[2] = z / length;
                  break;
               default:
                  std::cerr << __FILE__ << ":" << __LINE__ << ":" << "ionosphere.geometry has to be 0, 1 or 2 with this grid shape." << std::endl;
                  abort();
            }
            // end of X
         }
      } else if (Parameters::ycells_ini == 1) {
         if (Parameters::zcells_ini == 1) {
            // Y,Z
            normalDirection[0] = xsign;
            // end of Y,Z
         } else {
            // Y
            switch(this->geometry) {
               case 0:
                  normalDirection[0] = DIAG2*xsign;
                  normalDirection[2] = DIAG2*zsign;
                  break;
               case 1:
                  if(fabs(x) == fabs(z)) {
                     normalDirection[0] = xsign*DIAG2;
                     normalDirection[2] = zsign*DIAG2;
                     break;
                  }
                  if(fabs(x) > (this->radius - dx)) {
                     normalDirection[0] = xsign;
                     break;
                  }
                  if(fabs(z) > (this->radius - dz)) {
                     normalDirection[2] = zsign;
                     break;
                  }
                  if(fabs(x) > (this->radius - 2.0*dx)) {
                     normalDirection[0] = xsign;
                     break;
                  }
                  if(fabs(z) > (this->radius - 2.0*dz)) {
                     normalDirection[2] = zsign;
                     break;
                  }
                  break;
               case 2:
               case 3:
                  length = sqrt(x*x + z*z);
                  normalDirection[0] = x / length;
                  normalDirection[2] = z / length;
                  break;
               default:
                  std::cerr << __FILE__ << ":" << __LINE__ << ":" << "ionosphere.geometry has to be 0, 1, 2 or 3 with this grid shape." << std::endl;
                  abort();
            }
            // end of Y
         }
      } else if (Parameters::zcells_ini == 1) {
         // Z
         switch(this->geometry) {
            case 0:
               normalDirection[0] = DIAG2*xsign;
               normalDirection[1] = DIAG2*ysign;
               break;
            case 1:
               if(fabs(x) == fabs(y)) {
                  normalDirection[0] = xsign*DIAG2;
                  normalDirection[1] = ysign*DIAG2;
                  break;
               }
               if(fabs(x) > (this->radius - dx)) {
                  normalDirection[0] = xsign;
                  break;
               }
               if(fabs(y) > (this->radius - dy)) {
                  normalDirection[1] = ysign;
                  break;
               }
               if(fabs(x) > (this->radius - 2.0*dx)) {
                  normalDirection[0] = xsign;
                  break;
               }
               if(fabs(y) > (this->radius - 2.0*dy)) {
                  normalDirection[1] = ysign;
                  break;
               }
               break;
            case 2:
               length = sqrt(x*x + y*y);
               normalDirection[0] = x / length;
               normalDirection[1] = y / length;
               break;
            default:
               std::cerr << __FILE__ << ":" << __LINE__ << ":" << "ionosphere.geometry has to be 0, 1 or 2 with this grid shape." << std::endl;
               abort();
         }
         // end of Z
      } else {
         // 3D
         switch(this->geometry) {
            case 0:
               normalDirection[0] = DIAG3*xsign;
               normalDirection[1] = DIAG3*ysign;
               normalDirection[2] = DIAG3*zsign;
               break;
            case 1:
               if(fabs(x) == fabs(y) && fabs(x) == fabs(z) && fabs(x) > this->radius - dx) {
                  normalDirection[0] = xsign*DIAG3;
                  normalDirection[1] = ysign*DIAG3;
                  normalDirection[2] = zsign*DIAG3;
                  break;
               }
               if(fabs(x) == fabs(y) && fabs(x) == fabs(z) && fabs(x) > this->radius - 2.0*dx) {
                  normalDirection[0] = xsign*DIAG3;
                  normalDirection[1] = ysign*DIAG3;
                  normalDirection[2] = zsign*DIAG3;
                  break;
               }
               if(fabs(x) == fabs(y) && fabs(x) > this->radius - dx && fabs(z) < this->radius - dz) {
                  normalDirection[0] = xsign*DIAG2;
                  normalDirection[1] = ysign*DIAG2;
                  normalDirection[2] = 0.0;
                  break;
               }
               if(fabs(y) == fabs(z) && fabs(y) > this->radius - dy && fabs(x) < this->radius - dx) {
                  normalDirection[0] = 0.0;
                  normalDirection[1] = ysign*DIAG2;
                  normalDirection[2] = zsign*DIAG2;
                  break;
               }
               if(fabs(x) == fabs(z) && fabs(x) > this->radius - dx && fabs(y) < this->radius - dy) {
                  normalDirection[0] = xsign*DIAG2;
                  normalDirection[1] = 0.0;
                  normalDirection[2] = zsign*DIAG2;
                  break;
               }
               if(fabs(x) == fabs(y) && fabs(x) > this->radius - 2.0*dx && fabs(z) < this->radius - 2.0*dz) {
                  normalDirection[0] = xsign*DIAG2;
                  normalDirection[1] = ysign*DIAG2;
                  normalDirection[2] = 0.0;
                  break;
               }
               if(fabs(y) == fabs(z) && fabs(y) > this->radius - 2.0*dy && fabs(x) < this->radius - 2.0*dx) {
                  normalDirection[0] = 0.0;
                  normalDirection[1] = ysign*DIAG2;
                  normalDirection[2] = zsign*DIAG2;
                  break;
               }
               if(fabs(x) == fabs(z) && fabs(x) > this->radius - 2.0*dx && fabs(y) < this->radius - 2.0*dy) {
                  normalDirection[0] = xsign*DIAG2;
                  normalDirection[1] = 0.0;
                  normalDirection[2] = zsign*DIAG2;
                  break;
               }
               if(fabs(x) > (this->radius - dx)) {
                  normalDirection[0] = xsign;
                  break;
               }
               if(fabs(y) > (this->radius - dy)) {
                  normalDirection[1] = ysign;
                  break;
               }
               if(fabs(z) > (this->radius - dz)) {
                  normalDirection[2] = zsign;
                  break;
               }
               if(fabs(x) > (this->radius - 2.0*dx)) {
                  normalDirection[0] = xsign;
                  break;
               }
               if(fabs(y) > (this->radius - 2.0*dy)) {
                  normalDirection[1] = ysign;
                  break;
               }
               if(fabs(z) > (this->radius - 2.0*dz)) {
                  normalDirection[2] = zsign;
                  break;
               }
               break;
            case 2:
               length = sqrt(x*x + y*y + z*z);
               normalDirection[0] = x / length;
               normalDirection[1] = y / length;
               normalDirection[2] = z / length;
               break;
            case 3:
               length = sqrt(x*x + z*z);
               normalDirection[0] = x / length;
               normalDirection[2] = z / length;
               break;
            default:
               std::cerr << __FILE__ << ":" << __LINE__ << ":" << "ionosphere.geometry has to be 0, 1, 2 or 3 with this grid shape." << std::endl;
               abort();
         }
         // end of 3D
      }

      return normalDirection;
   }

   /*! We want here to
    *
    * -- Average perturbed face B from the nearest neighbours
    *
    * -- Retain only the normal components of perturbed face B
    */
   Real Ionosphere::fieldSolverBoundaryCondMagneticField(
      FsGrid< std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH> & bGrid,
      FsGrid< std::array<Real, fsgrids::bgbfield::N_BGB>, FS_STENCIL_WIDTH> & bgbGrid,
      FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
      cint i,
      cint j,
      cint k,
      creal dt,
      cuint component
   ) {
      if (technicalGrid.get(i,j,k)->sysBoundaryLayer == 1) {
         switch(component) {
            case 0:
               if (  ((technicalGrid.get(i-1,j,k)->SOLVE & compute::BX) == compute::BX)
                  && ((technicalGrid.get(i+1,j,k)->SOLVE & compute::BX) == compute::BX)
               ) {
                  return 0.5 * (bGrid.get(i-1,j,k)->at(fsgrids::bfield::PERBX) + bGrid.get(i+1,j,k)->at(fsgrids::bfield::PERBX));
               } else if ((technicalGrid.get(i-1,j,k)->SOLVE & compute::BX) == compute::BX) {
                  return bGrid.get(i-1,j,k)->at(fsgrids::bfield::PERBX);
               } else if ((technicalGrid.get(i+1,j,k)->SOLVE & compute::BX) == compute::BX) {
                  return bGrid.get(i+1,j,k)->at(fsgrids::bfield::PERBX);
               } else {
                  Real retval = 0.0;
                  uint nCells = 0;
                  if ((technicalGrid.get(i,j-1,k)->SOLVE & compute::BX) == compute::BX) {
                     retval += bGrid.get(i,j-1,k)->at(fsgrids::bfield::PERBX);
                     nCells++;
                  }
                  if ((technicalGrid.get(i,j+1,k)->SOLVE & compute::BX) == compute::BX) {
                     retval += bGrid.get(i,j+1,k)->at(fsgrids::bfield::PERBX);
                     nCells++;
                  }
                  if ((technicalGrid.get(i,j,k-1)->SOLVE & compute::BX) == compute::BX) {
                     retval += bGrid.get(i,j,k-1)->at(fsgrids::bfield::PERBX);
                     nCells++;
                  }
                  if ((technicalGrid.get(i,j,k+1)->SOLVE & compute::BX) == compute::BX) {
                     retval += bGrid.get(i,j,k+1)->at(fsgrids::bfield::PERBX);
                     nCells++;
                  }
                  if (nCells == 0) {
                     for (int a=i-1; a<i+2; a++) {
                        for (int b=j-1; b<j+2; b++) {
                           for (int c=k-1; c<k+2; c++) {
                              if ((technicalGrid.get(a,b,c)->SOLVE & compute::BX) == compute::BX) {
                                 retval += bGrid.get(a,b,c)->at(fsgrids::bfield::PERBX);
                                 nCells++;
                              }
                           }
                        }
                     }
                  }
                  if (nCells == 0) {
                     cerr << __FILE__ << ":" << __LINE__ << ": ERROR: this should not have fallen through." << endl;
                     return 0.0;
                  }
                  return retval / nCells;
               }
            case 1:
               if (  (technicalGrid.get(i,j-1,k)->SOLVE & compute::BY) == compute::BY
                  && (technicalGrid.get(i,j+1,k)->SOLVE & compute::BY) == compute::BY
               ) {
                  return 0.5 * (bGrid.get(i,j-1,k)->at(fsgrids::bfield::PERBY) + bGrid.get(i,j+1,k)->at(fsgrids::bfield::PERBY));
               } else if ((technicalGrid.get(i,j-1,k)->SOLVE & compute::BY) == compute::BY) {
                  return bGrid.get(i,j-1,k)->at(fsgrids::bfield::PERBY);
               } else if ((technicalGrid.get(i,j+1,k)->SOLVE & compute::BY) == compute::BY) {
                  return bGrid.get(i,j+1,k)->at(fsgrids::bfield::PERBY);
               } else {
                  Real retval = 0.0;
                  uint nCells = 0;
                  if ((technicalGrid.get(i-1,j,k)->SOLVE & compute::BY) == compute::BY) {
                     retval += bGrid.get(i-1,j,k)->at(fsgrids::bfield::PERBY);
                     nCells++;
                  }
                  if ((technicalGrid.get(i+1,j,k)->SOLVE & compute::BY) == compute::BY) {
                     retval += bGrid.get(i+1,j,k)->at(fsgrids::bfield::PERBY);
                     nCells++;
                  }
                  if ((technicalGrid.get(i,j,k-1)->SOLVE & compute::BY) == compute::BY) {
                     retval += bGrid.get(i,j,k-1)->at(fsgrids::bfield::PERBY);
                     nCells++;
                  }
                  if ((technicalGrid.get(i,j,k+1)->SOLVE & compute::BY) == compute::BY) {
                     retval += bGrid.get(i,j,k+1)->at(fsgrids::bfield::PERBY);
                     nCells++;
                  }
                  if (nCells == 0) {
                     for (int a=i-1; a<i+2; a++) {
                        for (int b=j-1; b<j+2; b++) {
                           for (int c=k-1; c<k+2; c++) {
                              if ((technicalGrid.get(a,b,c)->SOLVE & compute::BY) == compute::BY) {
                                 retval += bGrid.get(a,b,c)->at(fsgrids::bfield::PERBY);
                                 nCells++;
                              }
                           }
                        }
                     }
                  }
                  if (nCells == 0) {
                     cerr << __FILE__ << ":" << __LINE__ << ": ERROR: this should not have fallen through." << endl;
                     return 0.0;
                  }
                  return retval / nCells;
               }
            case 2:
               if (  (technicalGrid.get(i,j,k-1)->SOLVE & compute::BZ) == compute::BZ
                  && (technicalGrid.get(i,j,k+1)->SOLVE & compute::BZ) == compute::BZ
               ) {
                  return 0.5 * (bGrid.get(i,j,k-1)->at(fsgrids::bfield::PERBZ) + bGrid.get(i,j,k+1)->at(fsgrids::bfield::PERBZ));
               } else if ((technicalGrid.get(i,j,k-1)->SOLVE & compute::BZ) == compute::BZ) {
                  return bGrid.get(i,j,k-1)->at(fsgrids::bfield::PERBZ);
               } else if ((technicalGrid.get(i,j,k+1)->SOLVE & compute::BZ) == compute::BZ) {
                  return bGrid.get(i,j,k+1)->at(fsgrids::bfield::PERBZ);
               } else {
                  Real retval = 0.0;
                  uint nCells = 0;
                  if ((technicalGrid.get(i-1,j,k)->SOLVE & compute::BZ) == compute::BZ) {
                     retval += bGrid.get(i-1,j,k)->at(fsgrids::bfield::PERBZ);
                     nCells++;
                  }
                  if ((technicalGrid.get(i+1,j,k)->SOLVE & compute::BZ) == compute::BZ) {
                     retval += bGrid.get(i+1,j,k)->at(fsgrids::bfield::PERBZ);
                     nCells++;
                  }
                  if ((technicalGrid.get(i,j-1,k)->SOLVE & compute::BZ) == compute::BZ) {
                     retval += bGrid.get(i,j-1,k)->at(fsgrids::bfield::PERBZ);
                     nCells++;
                  }
                  if ((technicalGrid.get(i,j+1,k)->SOLVE & compute::BZ) == compute::BZ) {
                     retval += bGrid.get(i,j+1,k)->at(fsgrids::bfield::PERBZ);
                     nCells++;
                  }
                  if (nCells == 0) {
                     for (int a=i-1; a<i+2; a++) {
                        for (int b=j-1; b<j+2; b++) {
                           for (int c=k-1; c<k+2; c++) {
                              if ((technicalGrid.get(a,b,c)->SOLVE & compute::BZ) == compute::BZ) {
                                 retval += bGrid.get(a,b,c)->at(fsgrids::bfield::PERBZ);
                                 nCells++;
                              }
                           }
                        }
                     }
                  }
                  if (nCells == 0) {
                     cerr << __FILE__ << ":" << __LINE__ << ": ERROR: this should not have fallen through." << endl;
                     return 0.0;
                  }
                  return retval / nCells;
               }
            default:
               cerr << "ERROR: ionosphere boundary tried to copy nonsensical magnetic field component " << component << endl;
               return 0.0;
         }
      } else { // L2 cells
         Real retval = 0.0;
         uint nCells = 0;
         for (int a=i-1; a<i+2; a++) {
            for (int b=j-1; b<j+2; b++) {
               for (int c=k-1; c<k+2; c++) {
                  if (technicalGrid.get(a,b,c)->sysBoundaryLayer == 1) {
                     retval += bGrid.get(a,b,c)->at(fsgrids::bfield::PERBX + component);
                     nCells++;
                  }
               }
            }
         }
         if (nCells == 0) {
            cerr << __FILE__ << ":" << __LINE__ << ": ERROR: this should not have fallen through." << endl;
            return 0.0;
         }
         return retval / nCells;
      }
   }

   void Ionosphere::fieldSolverBoundaryCondElectricField(
      FsGrid< std::array<Real, fsgrids::efield::N_EFIELD>, FS_STENCIL_WIDTH> & EGrid,
      cint i,
      cint j,
      cint k,
      cuint component
   ) {
      EGrid.get(i,j,k)->at(fsgrids::efield::EX+component) = 0.0;
   }

   void Ionosphere::fieldSolverBoundaryCondHallElectricField(
      FsGrid< std::array<Real, fsgrids::ehall::N_EHALL>, FS_STENCIL_WIDTH> & EHallGrid,
      cint i,
      cint j,
      cint k,
      cuint component
   ) {
      std::array<Real, fsgrids::ehall::N_EHALL> * cp = EHallGrid.get(i,j,k);
      switch (component) {
         case 0:
            cp->at(fsgrids::ehall::EXHALL_000_100) = 0.0;
            cp->at(fsgrids::ehall::EXHALL_010_110) = 0.0;
            cp->at(fsgrids::ehall::EXHALL_001_101) = 0.0;
            cp->at(fsgrids::ehall::EXHALL_011_111) = 0.0;
            break;
         case 1:
            cp->at(fsgrids::ehall::EYHALL_000_010) = 0.0;
            cp->at(fsgrids::ehall::EYHALL_100_110) = 0.0;
            cp->at(fsgrids::ehall::EYHALL_001_011) = 0.0;
            cp->at(fsgrids::ehall::EYHALL_101_111) = 0.0;
            break;
         case 2:
            cp->at(fsgrids::ehall::EZHALL_000_001) = 0.0;
            cp->at(fsgrids::ehall::EZHALL_100_101) = 0.0;
            cp->at(fsgrids::ehall::EZHALL_010_011) = 0.0;
            cp->at(fsgrids::ehall::EZHALL_110_111) = 0.0;
            break;
         default:
            cerr << __FILE__ << ":" << __LINE__ << ":" << " Invalid component" << endl;
      }
   }

   void Ionosphere::fieldSolverBoundaryCondGradPeElectricField(
      FsGrid< std::array<Real, fsgrids::egradpe::N_EGRADPE>, FS_STENCIL_WIDTH> & EGradPeGrid,
      cint i,
      cint j,
      cint k,
      cuint component
   ) {
      EGradPeGrid.get(i,j,k)->at(fsgrids::egradpe::EXGRADPE+component) = 0.0;
   }

   void Ionosphere::fieldSolverBoundaryCondDerivatives(
      FsGrid< std::array<Real, fsgrids::dperb::N_DPERB>, FS_STENCIL_WIDTH> & dPerBGrid,
      FsGrid< std::array<Real, fsgrids::dmoments::N_DMOMENTS>, FS_STENCIL_WIDTH> & dMomentsGrid,
      cint i,
      cint j,
      cint k,
      cuint RKCase,
      cuint component
   ) {
      this->setCellDerivativesToZero(dPerBGrid, dMomentsGrid, i, j, k, component);
      return;
   }

   void Ionosphere::fieldSolverBoundaryCondBVOLDerivatives(
      FsGrid< std::array<Real, fsgrids::volfields::N_VOL>, FS_STENCIL_WIDTH> & volGrid,
      cint i,
      cint j,
      cint k,
      cuint component
   ) {
      // FIXME This should be OK as the BVOL derivatives are only used for Lorentz force JXB, which is not applied on the ionosphere cells.
      this->setCellBVOLDerivativesToZero(volGrid, i, j, k, component);
   }

   void Ionosphere::mapCellPotentialAndGetEXBDrift(
      std::array<Real, CellParams::N_SPATIAL_CELL_PARAMS>& cellParams
   ) {
      // Get potential upmapped from six points
      // (Cell's face centres)
      // inside the cell to calculate E
      const Real xmin = cellParams[CellParams::XCRD];
      const Real ymin = cellParams[CellParams::YCRD];
      const Real zmin = cellParams[CellParams::ZCRD];
      const Real xmax = xmin + cellParams[CellParams::DX];
      const Real ymax = ymin + cellParams[CellParams::DY];
      const Real zmax = zmin + cellParams[CellParams::DZ];
      const Real xcen = 0.5*(xmin+xmax);
      const Real ycen = 0.5*(ymin+ymax);
      const Real zcen = 0.5*(zmin+zmax);
      std::array< std::array<Real, 3>, 6> tracepoints;
      tracepoints[0] = {xmin, ycen, zcen};
      tracepoints[1] = {xmax, ycen, zcen};
      tracepoints[2] = {xcen, ymin, zcen};
      tracepoints[3] = {xcen, ymax, zcen};
      tracepoints[4] = {xcen, ycen, zmin};
      tracepoints[5] = {xcen, ycen, zmax};
      std::array<Real, 6> potentials;
       for(int i=0; i<6; i++) {
         // Get potential at each of these 6 points
         potentials[i] = ionosphereGrid.interpolateUpmappedPotential(tracepoints[i]);
      }

      // Calculate E from potential differences as E = -grad(phi)
      Vec3d E({
         (potentials[0] - potentials[1]) / cellParams[CellParams::DX],
         (potentials[2] - potentials[3]) / cellParams[CellParams::DY],
         (potentials[4] - potentials[5]) / cellParams[CellParams::DZ]});
      Vec3d B({
         cellParams[CellParams::BGBXVOL] + cellParams[CellParams::PERBXVOL],
         cellParams[CellParams::BGBYVOL] + cellParams[CellParams::PERBYVOL],
         cellParams[CellParams::BGBZVOL] + cellParams[CellParams::PERBZVOL]});

      // Add E from neutral wind convection for all cells with L <= 5
      Vec3d Omega(0,0,Ionosphere::earthAngularVelocity); // Earth rotation vector
      Vec3d r(xcen,ycen,zcen);
      Vec3d vn = cross_product(Omega,r);

      Real radius = vector_length(r);
      if(radius/physicalconstants::R_E <= Ionosphere::plasmapauseL * (r[0]*r[0] + r[1]*r[1]) / (radius*radius)) {
         E -= cross_product(vn, B);
      }

      const Real Bsqr = B[0]*B[0] + B[1]*B[1] + B[2]*B[2];

      // Calculate cell bulk velocity as E x B / B^2
      cellParams[CellParams::BULKV_FORCING_X] = (E[1] * B[2] - E[2] * B[1])/Bsqr;
      cellParams[CellParams::BULKV_FORCING_Y] = (E[2] * B[0] - E[0] * B[2])/Bsqr;
      cellParams[CellParams::BULKV_FORCING_Z] = (E[0] * B[1] - E[1] * B[0])/Bsqr;
   }

   void Ionosphere::vlasovBoundaryCondition(
      dccrg::Dccrg<SpatialCell,dccrg::Cartesian_Geometry>& mpiGrid,
      const CellID& cellID,
      const uint popID,
      const bool calculate_V_moments
   ) {
      // TODO Make this a more elegant solution
      // Now it's hacky as the counter is incremented in vlasiator.cpp
      if(globalflags::ionosphereJustSolved) { // else we don't update this boundary

         // If we are to couple to the ionosphere grid, we better be part of its communicator.
         assert(ionosphereGrid.communicator != MPI_COMM_NULL);

         mapCellPotentialAndGetEXBDrift(mpiGrid[cellID]->parameters);
         std::array<Real, 3> vDrift = {
            mpiGrid[cellID]->parameters[CellParams::BULKV_FORCING_X],
            mpiGrid[cellID]->parameters[CellParams::BULKV_FORCING_Y],
            mpiGrid[cellID]->parameters[CellParams::BULKV_FORCING_Z]
         };

         // Select representative moments for the VDFs
         Real temperature = 0;
         Real density = 0;
         switch(boundaryVDFmode) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
            case FixedMoments:
               density = speciesParams[popID].rho;
               temperature = speciesParams[popID].T;
               break;
            case AverageAllMoments:// Fall through (handled by if further down)
            case AverageMoments:
               // Maxwellian VDF boundary modes
               {
                  Real pressure = 0, vx = 0, vy = 0, vz = 0;
                  // Average density and temperature from the nearest cells
                  const vector<CellID>& closestCells = getAllClosestNonsysboundaryCells(cellID);
                  for (CellID celli : closestCells) {
                     density += mpiGrid[celli]->parameters[CellParams::RHOM];
                     pressure += mpiGrid[celli]->parameters[CellParams::P_11] + mpiGrid[celli]->parameters[CellParams::P_22] + mpiGrid[celli]->parameters[CellParams::P_33];
                     vx += mpiGrid[celli]->parameters[CellParams::VX];
                     vy += mpiGrid[celli]->parameters[CellParams::VY];
                     vz += mpiGrid[celli]->parameters[CellParams::VZ];
                  }
                  density /= closestCells.size()*physicalconstants::MASS_PROTON;
                  vx /= closestCells.size();
                  vy /= closestCells.size();
                  vz /= closestCells.size();
                  pressure /= 3.0*closestCells.size();
                  // TODO make this multipop
                  temperature = pressure / (density * physicalconstants::K_B);

                  if(boundaryVDFmode == AverageAllMoments) {
                     vDrift[0] += vx;
                     vDrift[1] += vy;
                     vDrift[2] += vz;
                  }
               }
               break;
            case CopyAndLosscone:
               // This is handled below
               break;
         }
#pragma GCC diagnostic pop

         // Fill velocity space
         switch(boundaryVDFmode) {
            case FixedMoments:
            case AverageAllMoments:
            case AverageMoments:
               {
                  // Fill velocity space with new maxwellian data
                  SpatialCell& cell = *mpiGrid[cellID];
                  cell.clear(popID,false); // Clear previous velocity space completely, do not de-allocate memory
                  creal initRho = density;
                  creal initT = temperature;
                  creal initV0X = vDrift[0];
                  creal initV0Y = vDrift[1];
                  creal initV0Z = vDrift[2];
                  creal mass = getObjectWrapper().particleSpecies[popID].mass;

                  // Find list of blocks to initialize.
                  const uint nRequested = SBC::findMaxwellianBlocksToInitialize(popID,cell, initRho, initT, initV0X, initV0Y, initV0Z);
                  // stores in vmesh->getGrid() (localToGlobalMap)
                  // with count in cell.get_population(popID).N_blocks

                  // Resize and populate mesh
                  cell.prepare_to_receive_blocks(popID);

                  // Set the reservation value (capacity is increased in add_velocity_blocks
                  const Realf minValue = cell.getVelocityBlockMinValue(popID);

                  // fills v-space into target

                  #ifdef USE_GPU
                  vmesh::VelocityMesh *vmesh = cell.dev_get_velocity_mesh(popID);
                  vmesh::VelocityBlockContainer* VBC = cell.dev_get_velocity_blocks(popID);
                  #else
                  vmesh::VelocityMesh *vmesh = cell.get_velocity_mesh(popID);
                  vmesh::VelocityBlockContainer* VBC = cell.get_velocity_blocks(popID);
                  #endif
                  // Loop over blocks
                  Realf rhosum = 0;
                  arch::parallel_reduce<arch::null>(
                     {WID, WID, WID, nRequested},
                     ARCH_LOOP_LAMBDA (const uint i, const uint j, const uint k, const uint initIndex, Realf *lsum ) {
                        vmesh::GlobalID *GIDlist = vmesh->getGrid()->data();
                        Realf* bufferData = VBC->getData();
                        const vmesh::GlobalID blockGID = GIDlist[initIndex];
                        // Calculate parameters for new block
                        Real blockCoords[6];
                        vmesh->getBlockInfo(blockGID,&blockCoords[0]);
                        creal vxBlock = blockCoords[0];
                        creal vyBlock = blockCoords[1];
                        creal vzBlock = blockCoords[2];
                        creal dvxCell = blockCoords[3];
                        creal dvyCell = blockCoords[4];
                        creal dvzCell = blockCoords[5];
                        ARCH_INNER_BODY(i, j, k, initIndex, lsum) {
                           creal vx = vxBlock + (i+0.5)*dvxCell - initV0X;
                           creal vy = vyBlock + (j+0.5)*dvyCell - initV0Y;
                           creal vz = vzBlock + (k+0.5)*dvzCell - initV0Z;
                           const Realf value = projects::MaxwellianPhaseSpaceDensity(vx,vy,vz,initT,initRho,mass);
                           bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
                           //lsum[0] += value;
                        };
                     }, rhosum);

                  #ifdef USE_GPU
                  // Set and apply the reservation value
                  cell.setReservation(popID,nRequested,true); // Force to this value
                  cell.applyReservation(popID);
                  #endif
               } // end case several
               break;
            case CopyAndLosscone:
               {
                  // GPUTODO: Untested after porting to new initialization
                  Real vNeighboursX = 0;
                  Real vNeighboursY = 0;
                  Real vNeighboursZ = 0;
                  Real pressure = 0;
                  // Get moments from the nearest cells
                  const vector<CellID>& closestCells = getAllClosestNonsysboundaryCells(cellID);
                  for (CellID celli : closestCells) {
                     density += mpiGrid[celli]->parameters[CellParams::RHOM];
                     pressure += mpiGrid[celli]->parameters[CellParams::P_11] + mpiGrid[celli]->parameters[CellParams::P_22] + mpiGrid[celli]->parameters[CellParams::P_33];
                     vNeighboursX += mpiGrid[celli]->parameters[CellParams::VX];
                     vNeighboursY += mpiGrid[celli]->parameters[CellParams::VY];
                     vNeighboursZ += mpiGrid[celli]->parameters[CellParams::VZ];
                  }
                  density /= closestCells.size()*physicalconstants::MASS_PROTON;
                  pressure /= 3.0*closestCells.size();
                  vNeighboursX /= closestCells.size();
                  vNeighboursY /= closestCells.size();
                  vNeighboursZ /= closestCells.size();
                  creal temperature = pressure / (density * physicalconstants::K_B);
                  // Fill velocity space with new VDF data. This consists of three parts:
                  // 1. For the downwards-moving part of the VDF (dot(v,r) < 0), simply fill a maxwellian with the averaged density and pressure.
                  // 2. For upwards-moving velocity cells outside the loss cone, take the reflected value from point 1.
                  // 3. Add an ionospheric outflow maxwellian.
                  SpatialCell& cell = *mpiGrid[cellID];
                  creal BX = cell.parameters[CellParams::BGBXVOL] + cell.parameters[CellParams::PERBXVOL];
                  creal BY = cell.parameters[CellParams::BGBYVOL] + cell.parameters[CellParams::PERBYVOL];
                  creal BZ = cell.parameters[CellParams::BGBZVOL] + cell.parameters[CellParams::PERBZVOL];
                  const Real Bsqr = BX*BX + BY*BY + BZ*BZ;
                  creal RX = cell.parameters[CellParams::XCRD] + 0.5*cell.parameters[CellParams::DX];
                  creal RY = cell.parameters[CellParams::YCRD] + 0.5*cell.parameters[CellParams::DY];
                  creal RZ = cell.parameters[CellParams::ZCRD] + 0.5*cell.parameters[CellParams::DZ];

                  cell.clear(popID,false); // Clear previous velocity space completely, do not de-allocate memory
                  creal initRho = speciesParams[popID].rho;
                  creal initT = speciesParams[popID].T;
                  creal initV0X = vDrift[0];
                  creal initV0Y = vDrift[1];
                  creal initV0Z = vDrift[2];
                  creal mass = getObjectWrapper().particleSpecies[popID].mass;

                  // Find list of blocks to initialize.
                  // WARNING: This now only finds blocks based on the outflow population, not including the copied losscone.
                  const uint nRequested = SBC::findMaxwellianBlocksToInitialize(popID,cell, initRho, initT, initV0X, initV0Y, initV0Z);
                  // stores in vmesh->getGrid() (localToGlobalMap)
                  // with count in cell.get_population(popID).N_blocks

                  // Resize and populate mesh
                  cell.prepare_to_receive_blocks(popID);

                  // Set the reservation value (capacity is increased in add_velocity_blocks
                  const Realf minValue = cell.getVelocityBlockMinValue(popID);

                  // fills v-space into target

                  #ifdef USE_GPU
                  vmesh::VelocityMesh *vmesh = cell.dev_get_velocity_mesh(popID);
                  vmesh::VelocityBlockContainer* VBC = cell.dev_get_velocity_blocks(popID);
                  #else
                  vmesh::VelocityMesh *vmesh = cell.get_velocity_mesh(popID);
                  vmesh::VelocityBlockContainer* VBC = cell.get_velocity_blocks(popID);
                  #endif
                  // Loop over blocks
                  Realf rhosum = 0;
                  arch::parallel_reduce<arch::null>(
                     {WID, WID, WID, nRequested},
                     ARCH_LOOP_LAMBDA (const uint i, const uint j, const uint k, const uint initIndex, Realf *lsum ) {
                        vmesh::GlobalID *GIDlist = vmesh->getGrid()->data();
                        Realf* bufferData = VBC->getData();
                        const vmesh::GlobalID blockGID = GIDlist[initIndex];
                        // Calculate parameters for new block
                        Real blockCoords[6];
                        vmesh->getBlockInfo(blockGID,&blockCoords[0]);
                        creal vxBlock = blockCoords[0];
                        creal vyBlock = blockCoords[1];
                        creal vzBlock = blockCoords[2];
                        creal dvxCell = blockCoords[3];
                        creal dvyCell = blockCoords[4];
                        creal dvzCell = blockCoords[5];
                        ARCH_INNER_BODY(i, j, k, initIndex, lsum) {
                           creal vx = vxBlock + (i+0.5)*dvxCell;
                           creal vy = vyBlock + (j+0.5)*dvyCell;
                           creal vz = vzBlock + (k+0.5)*dvzCell;

                           // Calculate pitchangle cosine
                           creal mu = (vx*BX + vy*BY + vz*BZ)/sqrt(Bsqr)/sqrt(vx*vx+vy*vy+vz*vz);
                           // Radial velocity component
                           creal rlength = sqrt(RX*RX + RY*RY + RZ*RZ);
                           creal RnormX = RX / rlength;
                           creal RnormY = RY / rlength;
                           creal RnormZ = RZ / rlength;
                           creal vdotr = (vx*RnormX + vy*RnormY * vz*RnormZ);

                           // v_r = -v_r = -r <v, r> (where r is normalized)
                           // => v = v - 2*r <r,v>
                           creal vNeighboursdotr = (vNeighboursX * RnormX + vNeighboursY * RnormY + vNeighboursZ * RnormZ);
                           creal vNeighboursMirroredX = vNeighboursX - 2*RnormX*vNeighboursdotr;
                           creal vNeighboursMirroredY = vNeighboursY - 2*RnormY*vNeighboursdotr;
                           creal vNeighboursMirroredZ = vNeighboursZ - 2*RnormZ*vNeighboursdotr;
                           Realf value = 0;
                           if (vdotr < 0) {
                              value = projects::MaxwellianPhaseSpaceDensity(vx - vNeighboursX,
                                                                            vy - vNeighboursY,
                                                                            vz - vNeighboursZ,
                                                                            temperature,density,mass);
                           } else {
                              if (1-mu*mu < sqrt(Bsqr)/5e-5) {
                                 // outside the loss cone
                                 value = projects::MaxwellianPhaseSpaceDensity(vx - 2*RnormX*vdotr - vNeighboursMirroredX,
                                                                               vy - 2*RnormY*vdotr - vNeighboursMirroredY,
                                                                               vz - 2*RnormZ*vdotr - vNeighboursMirroredZ,
                                                                               temperature,density,mass);
                              } else {
                                 // Inside the loss cone
                                 value = 0;
                              }
                           }
                           // Add ionospheric outflow maxwellian on top.
                           value += projects::MaxwellianPhaseSpaceDensity(vx - initV0X,
                                                                          vy - initV0Y,
                                                                          vz - initV0Z,
                                                                          initT,initRho,mass);
                           bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
                           //lsum[0] += value;
                        };
                     }, rhosum);

                  #ifdef USE_GPU
                  // Set and apply the reservation value
                  cell.setReservation(popID,nRequested,true); // Force to this value
                  cell.applyReservation(popID);
                  #endif
               } // end case CopyAndLosscone
               break;
         } // end switch VDF method
         // let's get rid of blocks not fulfilling the criteria here to save memory.
         mpiGrid[cellID]->adjustSingleCellVelocityBlocks(popID,true);

         // In principle this could call _R or _V instead according to calculate_V_moments (unused at the moment)
         // But the relevant moments will get recomputed in other spots when needed.
         calculateCellMoments(mpiGrid[cellID], true, false, true);
      } // End of if for coupling interval, we skip this altogether

   }

   /**
    * NOTE: This function must initialize all particle species!
    * @param project
    */
   void Ionosphere::generateTemplateCell(Project &project) {
      // WARNING not 0.0 here or the dipole() function fails miserably.
      templateCell.sysBoundaryFlag = this->getIndex();
      templateCell.sysBoundaryLayer = 1;
      templateCell.parameters[CellParams::XCRD] = 1.0;
      templateCell.parameters[CellParams::YCRD] = 1.0;
      templateCell.parameters[CellParams::ZCRD] = 1.0;
      templateCell.parameters[CellParams::DX] = 1;
      templateCell.parameters[CellParams::DY] = 1;
      templateCell.parameters[CellParams::DZ] = 1;

      Real initRho, initT, initV0X, initV0Y, initV0Z;
      // Loop over particle species
      for (uint popID=0; popID<getObjectWrapper().particleSpecies.size(); ++popID) {
         templateCell.clear(popID,false); //clear, do not de-allocate memory
         const IonosphereSpeciesParameters& sP = this->speciesParams[popID];
         const Real mass = getObjectWrapper().particleSpecies[popID].mass;
         initRho = sP.rho;
         initT = sP.T;
         initV0X = 0;
         initV0Y = 0;
         initV0Z = 0;

         // Find list of blocks to initialize.
         const uint nRequested = SBC::findMaxwellianBlocksToInitialize(popID,templateCell, initRho, initT, initV0X, initV0Y, initV0Z);
         // stores in vmesh->getGrid() (localToGlobalMap)
         // with count in cell.get_population(popID).N_blocks

         // Resize and populate mesh
         templateCell.prepare_to_receive_blocks(popID);

         // Set the reservation value (capacity is increased in add_velocity_blocks
         const Realf minValue = templateCell.getVelocityBlockMinValue(popID);

         // fills v-space into target

         #ifdef USE_GPU
         vmesh::VelocityMesh *vmesh = templateCell.dev_get_velocity_mesh(popID);
         vmesh::VelocityBlockContainer* VBC = templateCell.dev_get_velocity_blocks(popID);
         #else
         vmesh::VelocityMesh *vmesh = templateCell.get_velocity_mesh(popID);
         vmesh::VelocityBlockContainer* VBC = templateCell.get_velocity_blocks(popID);
         #endif
         // Loop over blocks
         Realf rhosum = 0;
         arch::parallel_reduce<arch::null>(
            {WID, WID, WID, nRequested},
            ARCH_LOOP_LAMBDA (const uint i, const uint j, const uint k, const uint initIndex, Realf *lsum ) {
               vmesh::GlobalID *GIDlist = vmesh->getGrid()->data();
               Realf* bufferData = VBC->getData();
               const vmesh::GlobalID blockGID = GIDlist[initIndex];
               // Calculate parameters for new block
               Real blockCoords[6];
               vmesh->getBlockInfo(blockGID,&blockCoords[0]);
               creal vxBlock = blockCoords[0];
               creal vyBlock = blockCoords[1];
               creal vzBlock = blockCoords[2];
               creal dvxCell = blockCoords[3];
               creal dvyCell = blockCoords[4];
               creal dvzCell = blockCoords[5];
               ARCH_INNER_BODY(i, j, k, initIndex, lsum) {
                  creal vx = vxBlock + (i+0.5)*dvxCell - initV0X;
                  creal vy = vyBlock + (j+0.5)*dvyCell - initV0Y;
                  creal vz = vzBlock + (k+0.5)*dvzCell - initV0Z;
                  const Realf value = projects::MaxwellianPhaseSpaceDensity(vx,vy,vz,initT,initRho,mass);
                  bufferData[initIndex*WID3 + k*WID2 + j*WID + i] = value;
                  //lsum[0] += value;
               };
            }, rhosum);

         #ifdef USE_GPU
         // Set and apply the reservation value
         templateCell.setReservation(popID,nRequested,true); // Force to this value
         templateCell.applyReservation(popID);
         #endif

         //let's get rid of blocks not fulfilling the criteria here to save memory.
         templateCell.adjustSingleCellVelocityBlocks(popID,true);
      } // for-loop over particle species

      calculateCellMoments(&templateCell,true,false,true);

      // WARNING Time-independence assumed here. Normal moments computed in setProjectCell
      templateCell.parameters[CellParams::RHOM_R] = templateCell.parameters[CellParams::RHOM];
      templateCell.parameters[CellParams::VX_R] = templateCell.parameters[CellParams::VX];
      templateCell.parameters[CellParams::VY_R] = templateCell.parameters[CellParams::VY];
      templateCell.parameters[CellParams::VZ_R] = templateCell.parameters[CellParams::VZ];
      templateCell.parameters[CellParams::RHOQ_R] = templateCell.parameters[CellParams::RHOQ];
      templateCell.parameters[CellParams::P_11_R] = templateCell.parameters[CellParams::P_11];
      templateCell.parameters[CellParams::P_22_R] = templateCell.parameters[CellParams::P_22];
      templateCell.parameters[CellParams::P_33_R] = templateCell.parameters[CellParams::P_33];
      templateCell.parameters[CellParams::RHOM_V] = templateCell.parameters[CellParams::RHOM];
      templateCell.parameters[CellParams::VX_V] = templateCell.parameters[CellParams::VX];
      templateCell.parameters[CellParams::VY_V] = templateCell.parameters[CellParams::VY];
      templateCell.parameters[CellParams::VZ_V] = templateCell.parameters[CellParams::VZ];
      templateCell.parameters[CellParams::RHOQ_V] = templateCell.parameters[CellParams::RHOQ];
      templateCell.parameters[CellParams::P_11_V] = templateCell.parameters[CellParams::P_11];
      templateCell.parameters[CellParams::P_22_V] = templateCell.parameters[CellParams::P_22];
      templateCell.parameters[CellParams::P_33_V] = templateCell.parameters[CellParams::P_33];
   }

   void Ionosphere::setCellFromTemplate(SpatialCell* cell,const uint popID) {
      copyCellData(&templateCell,cell,false,popID,true); // copy also vdf, _V
      copyCellData(&templateCell,cell,true,popID,false); // don't copy vdf again but copy _R now
      #ifdef USE_GPU
      cell->setReservation(popID,templateCell.getReservation(popID));
      #endif
   }

   std::string Ionosphere::getName() const {return "Ionosphere";}
   void Ionosphere::getFaces(bool *faces) {}

   void Ionosphere::updateState(dccrg::Dccrg<SpatialCell, dccrg::Cartesian_Geometry>& mpiGrid,
                                FsGrid< fsgrids::technical, FS_STENCIL_WIDTH> & technicalGrid,
                                FsGrid<std::array<Real, fsgrids::bfield::N_BFIELD>, FS_STENCIL_WIDTH>& perBGrid,
                                FsGrid<std::array<Real, fsgrids::bgbfield::N_BGB>, FS_STENCIL_WIDTH>& BgBGrid,
                                creal t) {}

   uint Ionosphere::getIndex() const {return sysboundarytype::IONOSPHERE;}
}
