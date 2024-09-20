/*
This file is part of CUDAProb3++.

CUDAProb3++ is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CUDAProb3++ is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with CUDAProb3++.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUDAPROB3_PROPAGATOR_HPP
#define CUDAPROB3_PROPAGATOR_HPP

#include "constants.hpp"
#include "types.hpp"
#include "math.hpp"


#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <iostream>



namespace cudaprob3{


    /// \class Propagator
    /// \brief Abstract base class of the library which sets up input parameter on the host.
    /// Concrete implementation of calcuations is provided in derived classes
    /// @param FLOAT_T The floating point type to use for calculations, i.e float, double
    template<class FLOAT_T>
    class Propagator{
    public:
        /// \brief Constructor (Atmospheric)
        ///
        /// @param n_cosines Number cosine bins
        /// @param num_energies Number of energy bins
        Propagator(int n_cosines, int num_energies) : n_cosines(n_cosines), n_energies(num_energies){
            energyList.resize(n_energies);
            cosineList.resize(n_cosines);
            maxlayers.resize(n_cosines);
        }

        /// \brief Constructor (Beam)
        ///
        /// @param n_energies Number of energy bins
        Propagator(int num_energies) : n_energies(num_energies){
            energyList.resize(n_energies);
        }

        /// \brief Copy constructor
        /// @param other
        Propagator(const Propagator& other){
            *this = other;
        }

        /// \brief Move constructor
        /// @param other
        Propagator(Propagator&& other){
            *this = std::move(other);
        }

        virtual ~Propagator(){}

        /// \brief Copy assignment operator
        /// @param other
     /*   Propagator& operator=(const Propagator& other){
            energyList = other.energyList;
            cosineList = other.cosineList;
            maxlayers = other.maxlayers;
            radii = other.radii;
            rhos = other.rhos;
            as = other.as;
            bs = other.bs;
            cs = other.cs;
	    yps = other.yps;
            coslimit = other.coslimit;
            Mix_U = other.Mix_U;
            dm = other.dm;

            ProductionHeightinCentimeter = other.ProductionHeightinCentimeter;
            isSetCosine = other.isSetCosine;
            isSetProductionHeight = other.isSetProductionHeight;
            isInit = other.isInit;

	    nProductionHeightBins = other.nProductionHeightBins;
	    useProductionHeightAveraging = other.useProductionHeightAveraging;

	    productionHeightList_prob = other.productionHeightList_prob;
	    productionHeightList_bins = other.productionHeightList_bins;
	    isSetProductionHeightArray = other.isSetProductionHeightArray;

            return *this;
        }

        /// \brief Move assignment operator
        /// @param other
        Propagator& operator=(Propagator&& other){
            energyList = std::move(other.energyList);
            cosineList = std::move(other.cosineList);
            maxlayers = std::move(other.maxlayers);
            radii = std::move(other.radii);
            rhos = std::move(other.rhos);
            as = std::move(other.as);
            bs = std::move(other.bs);
            cs = std::move(other.cs);
	    yps = std::move(other.yps);
            coslimit = std::move(other.coslimit);
            Mix_U = std::move(other.Mix_U);
            dm = std::move(other.dm);

            ProductionHeightinCentimeter = other.ProductionHeightinCentimeter;
            isSetCosine = other.isSetCosine;
            isSetProductionHeight = other.isSetProductionHeight;
            isInit = other.isInit;

	    nProductionHeightBins = other.nProductionHeightBins;
	    useProductionHeightAveraging = other.useProductionHeightAveraging;

            productionHeightList_prob = other.productionHeightList_prob;
            productionHeightList_bins = other.productionHeightList_bins;
            isSetProductionHeightArray = other.isSetProductionHeightArray;

            other.isInit = false;

            return *this;
        } */

    public:
      /// \brief Set mixing angles and cp phase in radians
      /// @param theta12
      /// @param theta13
      /// @param theta23
      /// @param dCP
      virtual void setMNSMatrix(FLOAT_T theta12, FLOAT_T theta13, FLOAT_T theta23, FLOAT_T dCP, int kNuType){

        if (kNuType < 0)
        {
          dCP *= -1;
        }

        const FLOAT_T s12 = sin(theta12);
        const FLOAT_T s13 = sin(theta13);
        const FLOAT_T s23 = sin(theta23);
        const FLOAT_T c12 = cos(theta12);
        const FLOAT_T c13 = cos(theta13);
        const FLOAT_T c23 = cos(theta23);

        const FLOAT_T sd  = sin(dCP);
        const FLOAT_T cd  = cos(dCP);

        U(0,0).re =  c12*c13;
        U(0,0).im =  0.0;
        U(0,1).re =  s12*c13;
        U(0,1).im =  0.0;
        U(0,2).re =  s13*cd;
        U(0,2).im = -s13*sd;
        U(1,0).re = -s12*c23-c12*s23*s13*cd;
        U(1,0).im =         -c12*s23*s13*sd;
        U(1,1).re =  c12*c23-s12*s23*s13*cd;
        U(1,1).im =         -s12*s23*s13*sd;
        U(1,2).re =  s23*c13;
        U(1,2).im =  0.0;
        U(2,0).re =  s12*s23-c12*c23*s13*cd;
        U(2,0).im =         -c12*c23*s13*sd;
        U(2,1).re = -c12*s23-s12*c23*s13*cd;
        U(2,1).im  =         -s12*c23*s13*sd;
        U(2,2).re =  c23*c13;
        U(2,2).im  =  0.0;
      }

      /// \brief Set neutrino mass differences (m_i_j)^2 in (eV)^2. no assumptions about mass hierarchy are made
      /// @param dm12sq
      /// @param dm23sq
      virtual void setNeutrinoMasses(FLOAT_T dm12sq, FLOAT_T dm23sq){
        FLOAT_T mVac[3];

        mVac[0] = 0.0;
        mVac[1] = dm12sq;
        mVac[2] = dm12sq + dm23sq;

        const FLOAT_T delta = 5.0e-9;
        /* Break any degeneracies */
        if (dm12sq == 0.0) mVac[0] -= delta;
        if (dm23sq == 0.0) mVac[2] += delta;

        DM(0,0) = 0.0;
        DM(1,1) = 0.0;
        DM(2,2) = 0.0;
        DM(0,1) = mVac[0]-mVac[1];
        DM(1,0) = -DM(0,1);
        DM(0,2) = mVac[0]-mVac[2];
        DM(2,0) = -DM(0,2);
        DM(1,2) = mVac[1]-mVac[2];
        DM(2,1) = -DM(1,2);
      }

      /// \brief Set the energy bins. Energies are given in GeV
      /// @param list Energy list
      virtual void setEnergyList(const std::vector<FLOAT_T>& list){
        if(list.size() != size_t(n_energies))
          throw std::runtime_error("Propagator::setEnergyList. Propagator was not created for this number of energy nodes");

        energyList = list;
      }

      /// \brief Set chemical composition of each layer in the Earth model
      /// \details Set chemical composition of each layer in the Earth model
      /// @param
      virtual void setChemicalComposition(const std::vector<FLOAT_T>& list) = 0;

      /// \brief Calculate the probability of each cell
      /// @param type Neutrino or Antineutrino
      virtual void calculateProbabilities(NeutrinoType type) = 0;

      /// \brief get oscillation weight for specific cosine and energy (ATMOSPHERIC)
      /// @param index_cosine Cosine bin index (zero based)
      /// @param index_energy Energy bin index (zero based)
      /// @param t Specify which probability P(i->j)
      //virtual FLOAT_T getProbability(int index_cosine, int index_energy, ProbType t) = 0;
      
      /// \brief get oscillation weight for specific energy (BEAM)
      /// @param index_energy Energy bin index (zero based)
      /// @param t Specify which probability P(i->j)
      //virtual FLOAT_T getProbability(int index_energy, ProbType t) = 0;

      /// \brief get oscillation weight for specific energy (ATMOS)
      /// @param index_energy Energy bin index (zero based)
      /// @param t Specify which probability P(i->j)
      //virtual FLOAT_T getProbability(int index_cosine, int index_energy, ProbType t) = 0;

      /// \brief get oscillation weight
      /// @param probArr Cosine bin index (zero based)
      /// @param t Specify which probability P(i->j)
      virtual void getProbabilityArr(FLOAT_T* probArr, ProbType t) = 0;

	  virtual void setPathLength( FLOAT_T path_length ) = 0;

	  virtual void setDensity( FLOAT_T rho ) = 0;

    protected:
      cudaprob3::math::ComplexNumber<FLOAT_T>& U(int i, int j){
        return Mix_U[( i * 3 + j)];
      }

      FLOAT_T& DM(int i, int j){
        return dm[( i * 3 + j)];
      }

      std::vector<FLOAT_T> energyList;
      std::vector<FLOAT_T> cosineList;
      std::vector<int> maxlayers;
      //std::vector<FLOAT_T> pathLengths;

      //std::vector<FLOAT_T> productionHeightList_prob;
      //std::vector<FLOAT_T> productionHeightList_bins;

      //std::vector<FLOAT_T> radii;
      //std::vector<FLOAT_T> rhos;
      //std::vector<FLOAT_T> as;
      //std::vector<FLOAT_T> bs;
      //std::vector<FLOAT_T> cs;
      std::vector<FLOAT_T> yps;
      //std::vector<FLOAT_T> coslimit;

      std::array<cudaprob3::math::ComplexNumber<FLOAT_T>, 9> Mix_U; // MNS mixing matrix
      std::array<FLOAT_T, 9> dm; // mass differences;

      //FLOAT_T ProductionHeightinCentimeter;
      FLOAT_T beam_density;
      FLOAT_T beam_path_length;

      bool useProductionHeightAveraging = false;
      int nProductionHeightBins = 0;

      bool isSetProductionHeightArray = false;
      bool isSetProductionHeight = false;
      bool isSetCosine = false;
      bool isInit = true;

      int n_cosines;
      int n_energies;
      int n_layers;

      // Use polynomial density for density averaging each track?
      bool UsePolyDensity;
    };





} // namespace cudaprob3


#endif
