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

#ifndef CUDAPROB3_CPUPROPAGATOR_HPP
#define CUDAPROB3_CPUPROPAGATOR_HPP

#include "constants.hpp"
#include "propagator.hpp"

#include <omp.h>
#include <vector>


namespace cudaprob3{

    /// \class CpuPropagator
    /// \brief Multi-threaded CPU neutrino propagation. Derived from Propagator
    /// @param FLOAT_T The floating point type to use for calculations, i.e float, double
    template<class FLOAT_T>
    class CpuPropagator : public Propagator<FLOAT_T>{
    public:
        /// \brief Constructor (Atmospheric)
        ///
        /// @param n_cosines Number cosine bins
        /// @param num_energies Number of energy bins
        /// @param threads Number of threads
        CpuPropagator(int num_cosines, int num_energies, int threads) : Propagator<FLOAT_T>(num_cosines, num_energies){

            omp_set_num_threads(threads);
        }

        /// \brief Constructor (Beam)
        ///
        /// @param num_energies Number of energy bins
        /// @param threads Number of threads
        CpuPropagator(int num_energies, int threads) : Propagator<FLOAT_T>(num_energies){

            omp_set_num_threads(threads);
        }

        /// \brief Copy constructor
        /// @param other
        CpuPropagator(const CpuPropagator& other) : Propagator<FLOAT_T>(other){
            *this = other;
        }

        /// \brief Move constructor
        /// @param other
        CpuPropagator(CpuPropagator&& other) : Propagator<FLOAT_T>(other){
            *this = std::move(other);
        }

        /// \brief Copy assignment operator
        /// @param other
        CpuPropagator& operator=(const CpuPropagator& other){
            Propagator<FLOAT_T>::operator=(other);

            return *this;
        }

        /// \brief Move assignment operator
        /// @param other
        CpuPropagator& operator=(CpuPropagator&& other){
            Propagator<FLOAT_T>::operator=(std::move(other));

            return *this;
        }

    public:

    private:
    };



}

#endif
