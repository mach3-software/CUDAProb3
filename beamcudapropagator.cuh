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

#include "constants.hpp"
#include "beamcpupropagator.hpp"
#include "physics.hpp"

//#ifdef __NVCC__ //change this to ifndef __NVCC__ before running doxygen. otherwise both classes are not included in the documentation
#ifdef GPU_ON

#ifndef CUDAPROB3_BEAMCUDAPROPAGATOR_CUH
#define CUDAPROB3_BEAMCUDAPROPAGATOR_CUH

#include "cuda_unique.cuh"

#include <algorithm>
#include <string>
#include <vector>


namespace cudaprob3{

    /// \class BeamCudaPropagatorSingle
    /// \brief Single-GPU beam neutrino propagation. Derived from Propagator
    /// @param FLOAT_T The floating point type to use for calculations, i.e float, double
    //template<class FLOAT_T>
    class BeamCudaPropagatorSingle : public BeamCpuPropagator<double> {
        template<typename>
        friend class CudaPropagator;
    public:

        /// \brief Constructor
        ///
        /// @param id device id of the GPU to use
        /// @param n_cosines_ Number cosine bins
        /// @param n_energies_ Number of energy bins
	/// @param n_threads_ Number of threads
        BeamCudaPropagatorSingle(int id, int n_energies_, int n_threads_); 

        /// \brief Constructor which uses device id 0
        ///
        /// @param n_energies Number of energy bins
        BeamCudaPropagatorSingle(int n_energiesm int n_threads); //: BeamCudaPropagatorSingle(0, n_energies, n_threads);

        /// \brief Destructor
        ~BeamCudaPropagatorSingle();

        BeamCudaPropagatorSingle(const BeamCudaPropagatorSingle& other) = delete;

        /// \brief Move constructor
        /// @param other
        BeamCudaPropagatorSingle(BeamCudaPropagatorSingle&& other); //: BeamCpuPropagator<double>(other);

        BeamCudaPropagatorSingle& operator=(const BeamCudaPropagatorSingle& other) = delete;

        /// \brief Move assignment operator
        /// @param other
        BeamCudaPropagatorSingle& operator=(BeamCudaPropagatorSingle&& other);

    public:

        void setDensity( double beam_density_) override;

        void setPathLength( double beam_path_length_) override;

        void setEnergyList(const std::vector<double>& list) override;

        // calculate the probability of each cell
        void calculateProbabilities(NeutrinoType type) override;

       // get oscillation weight for specific cosine and energy
        double getProbability(int index_energy, ProbType t) override;

        // get oscillation weight for specific energy
        void getProbabilityArr(double* probArr, ProbType t) override; 


    protected:

        // launch the calculation kernel without waiting for its completion
        void calculateBeamProbabilitiesAsync(NeutrinoType type);

        // wait for calculateProbabilitiesAsync to finish
        void waitForCompletion();

        // copy results from device to host
        void getResultFromDevice();

    private:
        unique_pinned_ptr<double> resultList;

        // density
        unique_dev_ptr<double> d_rhos;

        // path length
        unique_dev_ptr<double> d_path_lengths;

        unique_dev_ptr<double> d_energy_list;
        unique_dev_ptr<double> d_cosine_list;

        shared_dev_ptr<double> d_result_list;

        cudaStream_t stream;
        int deviceId;

        bool resultsResideOnHost = false;
    };

// Mutli-GPU class commented out for now
/*
    /// \class CudaPropagator
    /// \brief Multi-GPU neutrino propagation. Derived from Propagator.
    /// \details This is essentially a wrapper around multiple CudaPropagatorSingle instances, one per used GPU
    /// Most of the setters and calculation functions simply call the appropriate function for each GPU
    /// @param FLOAT_T The floating point type to use for calculations, i.e float, double
    template<class FLOAT_T>
      class CudaPropagator : public Propagator<FLOAT_T>{
        public:
          /// \brief Single GPU constructor for device id 0
          ///
          /// @param nc Number cosine bins
          /// @param ne Number of energy bins
          CudaPropagator(int nc, int ne) : CudaPropagator(std::vector<int>{0}, nc, ne, true){}

          /// \brief Constructor
          ///
          /// @param ids List of device ids of the GPUs to use
          /// @param nc Number cosine bins
          /// @param ne Number of energy bins
          /// @param failOnInvalidId If true, throw exception if ids contains an invalid device id
          CudaPropagator(const std::vector<int>& ids, int nc, int ne, bool failOnInvalidId = true) : Propagator<FLOAT_T>(nc, ne) {

            int nDevices;
            cudaGetDeviceCount(&nDevices);

            if(nDevices == 0) throw std::runtime_error("No GPU found");

            for(const auto& id: ids){
              if(id >= nDevices){
                if(failOnInvalidId){
                  std::cout << "Available GPUs:" << std::endl;
                  for(int j = 0; j < nDevices; j++){
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, j);
                    std::cout << "Id " << j << " : " << prop.name << std::endl;
                  }
                  throw std::runtime_error("The requested GPU Id " + std::to_string(id) + " is not available.");
                }else{
                  std::cout << "invalid device id found : " << id << std::endl;
                }
              }else{
                deviceIds.push_back(id);
              }
            }

            if(deviceIds.size() == 0){
              throw std::runtime_error("No valid device id found.");
            }
            cosineIndices.resize(deviceIds.size());
            localCosineIndices.resize(this->n_cosines);

            for(int icos = 0; icos < this->n_cosines; icos++){

              int deviceIndex = getCosineDeviceIndex(icos);

              cosineIndices[deviceIndex].push_back(icos);
              // the icos-th path is processed by GPU deviceIndex.
              // In the subproblem processed by GPU deviceIndex, the icos-th path is the localCosineIndices[icos]-th path
              localCosineIndices[icos] = cosineIndices[deviceIndex].size() - 1;
            }

            for(size_t i = 0; i < deviceIds.size() && i < size_t(this->n_cosines); i++){
              propagatorVector.push_back(
                  std::unique_ptr<CudaPropagatorSingle<FLOAT_T>>(
                    new CudaPropagatorSingle<FLOAT_T>(deviceIds[i], cosineIndices[i].size(), this->n_energies)
                    )
                  );
            }
          }

          CudaPropagator(const CudaPropagator& other) = delete;

          /// \brief Move constructor
          /// @param other
          CudaPropagator(CudaPropagator&& other) : Propagator<FLOAT_T>(other){
            *this = std::move(other);
          }

          CudaPropagator& operator=(const CudaPropagator& other) = delete;

          /// \brief Move assignment operator
          /// @param other
          CudaPropagator& operator=(CudaPropagator&& other){
            Propagator<FLOAT_T>::operator=(std::move(other));

            deviceIds = std::move(other.deviceIds);
            cosineIndices = std::move(other.cosineIndices);
            localCosineIndices = std::move(other.localCosineIndices);
            cosineBatches = std::move(other.cosineBatches);
            propagatorVector = std::move(other.propagatorVector);

            return *this;
          }

        public:

          void setDensityFromFile(const std::string& filename) override{
            Propagator<FLOAT_T>::setDensityFromFile(filename);

            for(auto& propagator : propagatorVector)
              propagator->setDensityFromFile(filename);
          }

          void setDensity(
              const std::vector<FLOAT_T>& radii, 
              const std::vector<FLOAT_T>& rhos, 
              const std::vector<FLOAT_T>& yps) override{
            Propagator<FLOAT_T>::setDensity(radii, rhos, yps);

            for(auto& propagator : propagatorVector)
              propagator->setDensity(radii, rhos, yps);
          }

          void setDensity(
              const std::vector<FLOAT_T>& radii, 
              const std::vector<FLOAT_T>& a, 
              const std::vector<FLOAT_T>& b, 
              const std::vector<FLOAT_T>& c, 
              const std::vector<FLOAT_T>& yps) override{
            Propagator<FLOAT_T>::setDensity(radii, a, b, c, yps);

            for(auto& propagator : propagatorVector)
              propagator->setDensity(radii, a, b, c, yps);
          }

          void setNeutrinoMasses(FLOAT_T dm12sq, FLOAT_T dm23sq) override{
            Propagator<FLOAT_T>::setNeutrinoMasses(dm12sq, dm23sq);

            for(auto& propagator : propagatorVector)
              propagator->setNeutrinoMasses(dm12sq, dm23sq);
          }

          void setMNSMatrix(FLOAT_T theta12, FLOAT_T theta13, FLOAT_T theta23, FLOAT_T dCP) override{
            Propagator<FLOAT_T>::setMNSMatrix(theta12, theta13, theta23, dCP);

            for(auto& propagator : propagatorVector)
              propagator->setMNSMatrix(theta12, theta13, theta23, dCP);
          }

          void setEnergyList(const std::vector<FLOAT_T>& list) override{
            Propagator<FLOAT_T>::setEnergyList(list);

            for(auto& propagator : propagatorVector)
              propagator->setEnergyList(list);
          }

          void setCosineList(const std::vector<FLOAT_T>& list) override{
            Propagator<FLOAT_T>::setCosineList(list);

            for(size_t i = 0; i < propagatorVector.size(); i++){
              // make list of cosines for GPU i and pass it to propagator i
              std::vector<FLOAT_T> myCos(cosineIndices[i].size());
              std::transform(cosineIndices[i].begin(),
                  cosineIndices[i].end(),
                  myCos.begin(),
                  [&](int icos){ return this->cosineList[icos]; }
                  );
              propagatorVector[i]->setCosineList(myCos);
            }
          }

          void setProductionHeight(FLOAT_T heightKM) override{
            Propagator<FLOAT_T>::setProductionHeight(heightKM);

            for(auto& propagator : propagatorVector)
              propagator->setProductionHeight(heightKM);
          }

        public:
          void calculateProbabilities(NeutrinoType type) override{

            for(auto& propagator : propagatorVector)
              propagator->calculateProbabilitiesAsync(type);

            for(auto& propagator : propagatorVector)
              propagator->waitForCompletion();
          }

          FLOAT_T getProbability(int index_cosine, int index_energy, ProbType t) override{
            const int deviceIndex = getCosineDeviceIndex(index_cosine);
            const int localCosineIndex = localCosineIndices[index_cosine];

            return propagatorVector[deviceIndex]->getProbability(localCosineIndex, index_energy, t);
          }

          void getProbabilityArr(FLOAT_T* probArr, ProbType t) {
            throw std::runtime_error("CudaPropagatorSingle::getProbabilityArr. Will not work!");
          }

        private:

          void setMaxlayers() override{
            Propagator<FLOAT_T>::setMaxlayers();

            for(auto& propagator : propagatorVector)
              propagator->setMaxlayers();
          }

          // get index in device id for the GPU which processes the index_cosine-th path
          int getCosineDeviceIndex(int index_cosine){
#if 0
            // block distribution
            int id = 0;
            for(int i = deviceIds.size(); i-- > 0;){
              if(index_cosine < (i+1) * n_cosines / deviceIds.size())
                id = i;
            }
#else
            // cyclic distribution.
            const int id = index_cosine % deviceIds.size();
#endif
            return id;
          }

        private:

          std::vector<int> deviceIds;
          std::vector<std::vector<int>> cosineIndices;
          std::vector<int> localCosineIndices;

          std::vector<int> cosineBatches;

          // one CudaPropagatorSingle per GPU
          std::vector<std::unique_ptr<CudaPropagatorSingle<FLOAT_T>>> propagatorVector;
      };

*/

}

#endif


//#endif // #ifdef __NVCC__
#endif // #ifdef GPU_ON
