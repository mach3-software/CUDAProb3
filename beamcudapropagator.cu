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

#ifdef __NVCC__  //change this to ifndef __NVCC__ before running doxygen. otherwise both classes are not included in the documentation

#ifndef CUDAPROB3_BEAMCUDAPROPAGATOR_CU
#define CUDAPROB3_BEAMCUDAPROPAGATOR_CU

#include "beamcudapropagator.cuh"

        /// \brief Constructor
        ///
        /// @param id device id of the GPU to use
        /// @param n_energies_ Number of energy bins
        cudaprob3linear::BeamCudaPropagatorSingle::BeamCudaPropagatorSingle(int id, int n_energies_, int n_threads_) : cudaprob3linear::BeamCpuPropagator<double>(n_energies_, n_threads_), deviceId(id){

            int nDevices;

            cudaGetDeviceCount(&nDevices); CUERR;

            if(nDevices == 0) throw std::runtime_error("No GPU found");
            if(id >= nDevices){
                std::cout << "Available GPUs:" << std::endl;
                for(int j = 0; j < nDevices; j++){
                    cudaDeviceProp prop;
                    cudaGetDeviceProperties(&prop, j); CUERR;
                    std::cout << "Id " << j << " : " << prop.name << std::endl;
                }
                throw std::runtime_error("The requested GPU Id " + std::to_string(id) + " is not available.");
            }

            cudaSetDevice(id); CUERR;
            cudaFree(0);

            cudaStreamCreate(&stream); CUERR;

            //allocate host arrays which are not already allocated by Propagator base class
            resultList = make_unique_pinned<double>(std::uint64_t(n_energies_) * std::uint64_t(9));

            //allocate GPU arrays
            d_energy_list = make_unique_dev<double>(deviceId, n_energies_); CUERR;
            d_result_list = make_shared_dev<double>(deviceId, std::uint64_t(n_energies_) * std::uint64_t(9)); CUERR;
            d_rhos = make_unique_dev<double>(deviceId, std::uint64_t(n_energies_));
            d_path_lengths = make_unique_dev<double>(deviceId, std::uint64_t(n_energies_));
        }

        /// \brief Constructor which uses device id 0
        ///
        /// @param n_energies Number of energy bins
        cudaprob3linear::BeamCudaPropagatorSingle::BeamCudaPropagatorSingle(int n_energies, int n_threads) : cudaprob3linear::BeamCudaPropagatorSingle::BeamCudaPropagatorSingle(0, n_energies, n_threads){

        }

        /// \brief Destructor
        cudaprob3linear::BeamCudaPropagatorSingle::~BeamCudaPropagatorSingle(){
            cudaSetDevice(deviceId);
            cudaStreamDestroy(stream);
        }

        /// \brief Move constructor
        /// @param other
        //cudaprob3linear::BeamCudaPropagatorSingle::BeamCudaPropagatorSingle(BeamCudaPropagatorSingle&& other) : BeamCpuPropagator<double>(other){
          //  *this = std::move(other);

          //  cudaSetDevice(deviceId);
          //  cudaStreamCreate(&stream); CUERR;
       // }

        //cudaprob3linear::BeamCudaPropagatorSingle::BeamCudaPropagatorSingle& operator=(const BeamCudaPropagatorSingle& other) = delete;

        /// \brief Move assignment operator
        /// @param other
       /* cudaprob3linear::BeamCudaPropagatorSingle::BeamCudaPropagatorSingle& operator=(BeamCudaPropagatorSingle&& other){
            cudaprob3linear::BeamCpuPropagator<double>::operator=(std::move(other));

            resultList = std::move(other.resultList);
            d_energy_list = std::move(other.d_energy_list);
            d_cosine_list = std::move(other.d_cosine_list);
            d_result_list = std::move(other.d_result_list);

            deviceId = other.deviceId;
            resultsResideOnHost = other.resultsResideOnHost;

            //the stream is not moved

            return *this;
        } */

        void cudaprob3linear::BeamCudaPropagatorSingle::setDensity( double beam_density_) {
          // call parent function to set up host density data
          BeamCpuPropagator<double>::setDensity(beam_density_);

          // allocate GPU arrays for density information and copy host density data to device density data
          cudaSetDevice(deviceId); CUERR;

          std::vector<double> rho(this->n_energies, this->beam_density);         
          //d_rhos = make_unique_dev<double>(deviceId, this->n_energies);

          //cudaMemcpy(rho, &this->beam_density, sizeof(double), H2D); CUERR;
          cudaMemcpy(d_rhos.get(), rho.data(), sizeof(double) * this->n_energies, H2D); CUERR;
        }

        void cudaprob3linear::BeamCudaPropagatorSingle::setPathLength( double beam_path_length_) {
          // call parent function to set up host density data
          BeamCpuPropagator<double>::setPathLength(beam_path_length_);

          // allocate GPU arrays for path length information and copy host density data to device density data
          cudaSetDevice(deviceId); CUERR;

          std::vector<double> len(this->n_energies, this->beam_path_length);         
          
          //d_path_lengths = make_unique_dev<double>(deviceId, this-n_energies);

          //cudaMemcpy(len, &this->beam_path_length, sizeof(double), H2D); CUERR;
          cudaMemcpy(d_path_lengths.get(), len.data(), sizeof(double) * this->n_energies, H2D); CUERR;
        }

        void cudaprob3linear::BeamCudaPropagatorSingle::setEnergyList(const std::vector<double>& list) {
          Propagator<double>::setEnergyList(list); // set host energy list

          //copy host energy list to gpu memory
          cudaMemcpy(d_energy_list.get(), this->energyList.data(), sizeof(double) * this->n_energies, H2D); CUERR;
        }

        // calculate the probability of each cell
        void cudaprob3linear::BeamCudaPropagatorSingle::calculateProbabilities(NeutrinoType type) {
          calculateBeamProbabilitiesAsync(type);
          waitForCompletion();
        }

        // get oscillation weight for specific cosine and energy
        double cudaprob3linear::BeamCudaPropagatorSingle::getProbability(int index_energy, ProbType t) {
          if(index_energy >= this->n_energies)
            throw std::runtime_error("CudaPropagatorSingle::getProbability. Invalid indices");

          if(!resultsResideOnHost){
            getResultFromDevice();
            resultsResideOnHost = true;
          }

          const std::uint64_t index = std::uint64_t(index_energy);
          const std::uint64_t offset = std::uint64_t(t) * std::uint64_t(this->n_energies);

          return resultList.get()[index + offset];
        }

        // get oscillation weight for specific energy
        void cudaprob3linear::BeamCudaPropagatorSingle::getProbabilityArr(double* probArr, ProbType t) {

          if(!resultsResideOnHost){
            getResultFromDevice();
            resultsResideOnHost = true;
          }

          double* resultList_Arr = resultList.get();
          const std::uint64_t offset = std::uint64_t(t) * std::uint64_t(this->n_energies);

          std::uint64_t iter = 0;
          for (int index_energy=0;index_energy<this->n_energies;index_energy++) {
              std::uint64_t index = std::uint64_t(index_energy);
              probArr[iter] = resultList_Arr[index + offset];
              iter += 1;
            }
        }

        // launch the calculation kernel without waiting for its completion
        void cudaprob3linear::BeamCudaPropagatorSingle::calculateBeamProbabilitiesAsync(NeutrinoType type){
          if(!this->isInit)
            throw std::runtime_error("CudaPropagatorSingle::calculateProbabilities. Object has been moved from.");

          resultsResideOnHost = false;
          cudaSetDevice(deviceId); CUERR;

          // set neutrino parameters for core physics functions for both host and device
          physics::setMixMatrix(this->Mix_U.data());
          physics::setMassDifferences(this->dm.data());

          dim3 block(64, 1, 1);

          //const unsigned blocks = SDIV(this->energyList.size(), block.x);
          const unsigned blocks = SDIV(this->energyList.size(), block.x);

          dim3 grid(blocks, 1, 1);
          physics::callCalculateBeamKernelAsync(grid, block, stream,
              type,
              d_energy_list.get(), 
              this->n_energies,
              this->beam_density, 
              this->beam_path_length,
              d_result_list.get());


          CUERR;
        }

        // wait for calculateProbabilitiesAsync to finish
        void cudaprob3linear::BeamCudaPropagatorSingle::waitForCompletion(){

          cudaSetDevice(deviceId); CUERR;
          cudaStreamSynchronize(stream); CUERR;
        }

        // copy results from device to host
        void cudaprob3linear::BeamCudaPropagatorSingle::getResultFromDevice(){
          cudaSetDevice(deviceId); CUERR;
          cudaMemcpyAsync(resultList.get(), d_result_list.get(),
              sizeof(double) * std::uint64_t(9) * std::uint64_t(this->n_energies),
              D2H, stream);  CUERR;
          cudaStreamSynchronize(stream);
        }

// Mutli-GPU class commented out for now
/*
    /// \class CudaPropagator
    /// \brief Multi-GPU neutrino propagation. Derived from Propagator.
    /// \details This is essentially a wrapper around multiple CudaPropagatorSingle instances, one per used GPU
    /// Most of the setters and calculation functions simply call the appropriate function for each GPU
    /// @param double The floating point type to use for calculations, i.e float, double
    template<class double>
      class CudaPropagator : public Propagator<double>{
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
          CudaPropagator(const std::vector<int>& ids, int nc, int ne, bool failOnInvalidId = true) : Propagator<double>(nc, ne) {

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
                  std::unique_ptr<CudaPropagatorSingle<double>>(
                    new CudaPropagatorSingle<double>(deviceIds[i], cosineIndices[i].size(), this->n_energies)
                    )
                  );
            }
          }

          CudaPropagator(const CudaPropagator& other) = delete;

          /// \brief Move constructor
          /// @param other
          CudaPropagator(CudaPropagator&& other) : Propagator<double>(other){
            *this = std::move(other);
          }

          CudaPropagator& operator=(const CudaPropagator& other) = delete;

          /// \brief Move assignment operator
          /// @param other
          CudaPropagator& operator=(CudaPropagator&& other){
            Propagator<double>::operator=(std::move(other));

            deviceIds = std::move(other.deviceIds);
            cosineIndices = std::move(other.cosineIndices);
            localCosineIndices = std::move(other.localCosineIndices);
            cosineBatches = std::move(other.cosineBatches);
            propagatorVector = std::move(other.propagatorVector);

            return *this;
          }

        public:

          void setDensityFromFile(const std::string& filename) override{
            Propagator<double>::setDensityFromFile(filename);

            for(auto& propagator : propagatorVector)
              propagator->setDensityFromFile(filename);
          }

          void setDensity(
              const std::vector<double>& radii, 
              const std::vector<double>& rhos, 
              const std::vector<double>& yps) override{
            Propagator<double>::setDensity(radii, rhos, yps);

            for(auto& propagator : propagatorVector)
              propagator->setDensity(radii, rhos, yps);
          }

          void setDensity(
              const std::vector<double>& radii, 
              const std::vector<double>& a, 
              const std::vector<double>& b, 
              const std::vector<double>& c, 
              const std::vector<double>& yps) override{
            Propagator<double>::setDensity(radii, a, b, c, yps);

            for(auto& propagator : propagatorVector)
              propagator->setDensity(radii, a, b, c, yps);
          }

          void setNeutrinoMasses(double dm12sq, double dm23sq) override{
            Propagator<double>::setNeutrinoMasses(dm12sq, dm23sq);

            for(auto& propagator : propagatorVector)
              propagator->setNeutrinoMasses(dm12sq, dm23sq);
          }

          void setMNSMatrix(double theta12, double theta13, double theta23, double dCP) override{
            Propagator<double>::setMNSMatrix(theta12, theta13, theta23, dCP);

            for(auto& propagator : propagatorVector)
              propagator->setMNSMatrix(theta12, theta13, theta23, dCP);
          }

          void setEnergyList(const std::vector<double>& list) override{
            Propagator<double>::setEnergyList(list);

            for(auto& propagator : propagatorVector)
              propagator->setEnergyList(list);
          }

          void setCosineList(const std::vector<double>& list) override{
            Propagator<double>::setCosineList(list);

            for(size_t i = 0; i < propagatorVector.size(); i++){
              // make list of cosines for GPU i and pass it to propagator i
              std::vector<double> myCos(cosineIndices[i].size());
              std::transform(cosineIndices[i].begin(),
                  cosineIndices[i].end(),
                  myCos.begin(),
                  [&](int icos){ return this->cosineList[icos]; }
                  );
              propagatorVector[i]->setCosineList(myCos);
            }
          }

          void setProductionHeight(double heightKM) override{
            Propagator<double>::setProductionHeight(heightKM);

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

          double getProbability(int index_cosine, int index_energy, ProbType t) override{
            const int deviceIndex = getCosineDeviceIndex(index_cosine);
            const int localCosineIndex = localCosineIndices[index_cosine];

            return propagatorVector[deviceIndex]->getProbability(localCosineIndex, index_energy, t);
          }

          void getProbabilityArr(double* probArr, ProbType t) {
            throw std::runtime_error("CudaPropagatorSingle::getProbabilityArr. Will not work!");
          }

        private:

          void setMaxlayers() override{
            Propagator<double>::setMaxlayers();

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
          std::vector<std::unique_ptr<CudaPropagatorSingle<double>>> propagatorVector;
      };

*/

//}

#endif

#endif // #ifdef __NVCC
