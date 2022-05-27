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

#ifndef CUDAPROB3_BEAMCPUPROPAGATOR_HPP
#define CUDAPROB3_BEAMCPUPROPAGATOR_HPP

#include "constants.hpp"
#include "propagator.hpp"
#include "cpupropagator.hpp"
#include "physics.hpp"

#include <omp.h>
#include <vector>


namespace cudaprob3{

    /// \class BeamCpuPropagator
    /// \brief Multi-threaded CPU neutrino propagation. Derived from CpuPropagator
    /// @param FLOAT_T The floating point type to use for calculations, i.e float, double
    template<class FLOAT_T>
    class BeamCpuPropagator : public CpuPropagator<FLOAT_T>{
    public:
        /// \brief Constructor
        ///
        /// @param n_cosines Number cosine bins
        /// @param n_energies Number of energy bins
        /// @param threads Number of threads
        BeamCpuPropagator(int n_energies, int threads) : CpuPropagator<FLOAT_T>(n_energies, threads){

            resultList.resize(std::uint64_t(n_energies) * std::uint64_t(9));

            omp_set_num_threads(threads);
        }

        /// \brief Copy constructor
        /// @param other
        BeamCpuPropagator(const BeamCpuPropagator& other) : CpuPropagator<FLOAT_T>(other){
            *this = other;
        }

        /// \brief Move constructor
        /// @param other
        BeamCpuPropagator(BeamCpuPropagator&& other) : CpuPropagator<FLOAT_T>(other){
            *this = std::move(other);
        }

        /// \brief Copy assignment operator
        /// @param other
        BeamCpuPropagator& operator=(const BeamCpuPropagator& other){
            CpuPropagator<FLOAT_T>::operator=(other);

            resultList = other.resultList;

            return *this;
        }

        /// \brief Move assignment operator
        /// @param other
        BeamCpuPropagator& operator=(BeamCpuPropagator&& other){
            CpuPropagator<FLOAT_T>::operator=(std::move(other));

            resultList = std::move(other.resultList);

            return *this;
        }

    public:

        virtual void setDensity( FLOAT_T rho ) {
	    beam_density = rho;
        }


        virtual void setPathLength( FLOAT_T path_length ) {
	    beam_path_length = path_length; 
        }

        virtual void calculateProbabilities(NeutrinoType type) override{

            if(!this->isInit)
                throw std::runtime_error("BeamCpuPropagator::calculateProbabilities. Object has been moved from.");

            // set neutrino parameters for core physics functions
            physics::setMixMatrix_host(this->Mix_U.data());
            physics::setMassDifferences_host(this->dm.data());

            physics::calculate_beam(type, 
                this->energyList.data(), 
                this->energyList.size(), 
                this->beam_density, 
                this->beam_path_length, 
                resultList.data());


        }

        virtual void setChemicalComposition(const std::vector<FLOAT_T>& list) override{
          if (list.size() != this->yps.size()) {
            throw std::runtime_error("BeamCpuPropagator::setChemicalComposition. Size of input list not equal to expectation");
          }

          for (std::uint64_t iyp=0;iyp<list.size();iyp++) {
            this->yps[iyp] = list[iyp];
          }

        }

        virtual FLOAT_T getProbability(int index_energy, ProbType t) {
          if(index_energy >= this->n_energies) {
            throw std::runtime_error("BeamCpuPropagator::getProbability. Invalid indices");
          }

          std::uint64_t index = std::uint64_t(index_energy) * std::uint64_t(9);
          return resultList[index + int(t)];
        }

        virtual void getProbabilityArr(FLOAT_T* probArr, ProbType t) {

          std::uint64_t iter = 0;
          for (int index_energy=0;index_energy<this->n_energies;index_energy++) {
            std::uint64_t index = std::uint64_t(index_energy) * std::uint64_t(9);
              probArr[iter] = resultList[index + int(t)];
              iter += 1;
          }
	}

    protected:

	FLOAT_T beam_path_length;	
	FLOAT_T beam_density;	

    private:
        std::vector<FLOAT_T> resultList;
    };



}

#endif
