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

#ifndef CUDAPROB3_ATMOSCUDAPROPAGATOR_CU
#define CUDAPROB3_ATMOSCUDAPROPAGATOR_CU

#include "atmoscudapropagator.cuh"

        /// \brief Constructor
        ///
        /// @param id device id of the GPU to use
        /// @param n_cosines_ Number cosine bins
        /// @param n_energies_ Number of energy bins
        cudaprob3::AtmosCudaPropagatorSingle::AtmosCudaPropagatorSingle(int id, int n_cosines_, int n_energies_) : cudaprob3::AtmosCpuPropagator<double>(n_cosines_, n_energies_, 1), deviceId(id){

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
            resultList = make_unique_pinned<double>(std::uint64_t(n_cosines_) * std::uint64_t(n_energies_) * std::uint64_t(9));

            //allocate GPU arrays
            d_energy_list = make_unique_dev<double>(deviceId, n_energies_); CUERR;
            d_cosine_list = make_unique_dev<double>(deviceId, n_cosines_); CUERR;
            d_productionHeight_prob_list = make_unique_dev<double>(deviceId, Constants<double>::MaxProdHeightBins()*2*3*n_energies_*n_cosines_); CUERR;
            d_productionHeight_bins_list = make_unique_dev<double>(deviceId, Constants<double>::MaxProdHeightBins()+1); CUERR;
            d_result_list = make_shared_dev<double>(deviceId, std::uint64_t(n_cosines_) * std::uint64_t(n_energies_) * std::uint64_t(9)); CUERR;
            d_maxlayers = make_unique_dev<int>(deviceId, this->n_cosines);
        }

        /// \brief Constructor which uses device id 0
        ///
        /// @param n_cosines Number cosine bins
        /// @param n_energies Number of energy bins
        cudaprob3::AtmosCudaPropagatorSingle::AtmosCudaPropagatorSingle(int n_cosines, int n_energies) : cudaprob3::AtmosCudaPropagatorSingle(0, n_cosines, n_energies){

        }

        /// \brief Destructor
        cudaprob3::AtmosCudaPropagatorSingle::~AtmosCudaPropagatorSingle(){
            cudaSetDevice(deviceId);
            cudaStreamDestroy(stream);
        }

        //cudaprob3::AtmosCudaPropagatorSingle::AtmosCudaPropagatorSingle(const AtmosCudaPropagatorSingle& other) = delete;

/*
        /// \brief Move constructor
        /// @param other
        AtmosCudaPropagatorSingle(AtmosCudaPropagatorSingle&& other) : AtmosCpuPropagator<double>(other){
            *this = std::move(other);

            cudaSetDevice(deviceId);
            cudaStreamCreate(&stream); CUERR;
        }

        AtmosCudaPropagatorSingle& operator=(const AtmosCudaPropagatorSingle& other) = delete;



        /// \brief Move assignment operator
        /// @param other
        AtmosCudaPropagatorSingle& operator=(AtmosCudaPropagatorSingle&& other){
            AtmosCpuPropagator<double>::operator=(std::move(other));

            resultList = std::move(other.resultList);
            d_rhos = std::move(other.d_rhos);
            d_as = std::move(other.d_as);
            d_bs = std::move(other.d_bs);
            d_cs = std::move(other.d_cs);
	    d_yps = std::move(other.d_yps);
            d_radii = std::move(other.d_radii);
            d_maxlayers = std::move(other.d_maxlayers);
            d_energy_list = std::move(other.d_energy_list);
            d_cosine_list = std::move(other.d_cosine_list);
            d_productionHeight_prob_list = std::move(other.d_productionHeight_prob_list);
            d_productionHeight_bins_list = std::move(other.d_productionHeight_bins_list);
            d_result_list = std::move(other.d_result_list);

            deviceId = other.deviceId;
            resultsResideOnHost = other.resultsResideOnHost;

            //the stream is not moved

            return *this;
        } */

        void cudaprob3::AtmosCudaPropagatorSingle::setDensity(
          const std::vector<double>& radii_, 
          const std::vector<double>& rhos_, 
          const std::vector<double>& yps_) {
          // call parent function to set up host density data
          AtmosCpuPropagator<double>::setDensity(radii_, rhos_, yps_);

          // allocate GPU arrays for density information and copy host density data to device density data
          cudaSetDevice(deviceId); CUERR;

          int nDensityLayers = this->radii.size();

          d_rhos = make_unique_dev<double>(deviceId, 2 * nDensityLayers + 1);
          d_yps = make_unique_dev<double>(deviceId, 2 * nDensityLayers + 1);
          d_radii = make_unique_dev<double>(deviceId, 2 * nDensityLayers + 1);

          cudaMemcpy(d_rhos.get(), this->rhos.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
          cudaMemcpy(d_yps.get(), this->yps.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
          cudaMemcpy(d_radii.get(), this->radii.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
        }

        void cudaprob3::AtmosCudaPropagatorSingle::setDensity(
          const std::vector<double>& radii_, 
          const std::vector<double>& as_,
          const std::vector<double>& bs_,
          const std::vector<double>& cs_,
          const std::vector<double>& yps_) {

          // call parent function to set up host density data
          AtmosCpuPropagator<double>::setDensity(radii_, as_, bs_, cs_, yps_);

          // allocate GPU arrays for density information and copy host density data to device density data
          cudaSetDevice(deviceId); CUERR;

          int nDensityLayers = this->radii.size();

          d_as = make_unique_dev<double>(deviceId, 2 * nDensityLayers + 1);
          d_bs = make_unique_dev<double>(deviceId, 2 * nDensityLayers + 1);
          d_cs = make_unique_dev<double>(deviceId, 2 * nDensityLayers + 1);
          d_yps = make_unique_dev<double>(deviceId, 2 * nDensityLayers + 1);
          d_radii = make_unique_dev<double>(deviceId, 2 * nDensityLayers + 1);

          cudaMemcpy(d_as.get(), this->as.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
          cudaMemcpy(d_bs.get(), this->bs.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
          cudaMemcpy(d_cs.get(), this->cs.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
          cudaMemcpy(d_yps.get(), this->yps.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
          cudaMemcpy(d_radii.get(), this->radii.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
        }

        void cudaprob3::AtmosCudaPropagatorSingle::setDensity( double rho ) {
	      std::cout << "DUMMY FUNCTION: ATMOS class uses setDensity( \n" ;
          std::cout << "const std::vector<double>& radii_, \n " ;
          std::cout << "const std::vector<double>& a_, \n " ;
          std::cout << "const std::vector<double>& b_, \n " ;
          std::cout << "const std::vector<double>& c_, \n " ;
          std::cout << "const std::vector<double>& yps_) \n " ;
		  std::cout << "or \n " ;
          std::cout << "setDensityFromFile(const std::string& filename) " << std::endl;
		}

        void cudaprob3::AtmosCudaPropagatorSingle::setEnergyList(const std::vector<double>& list) {
          AtmosCpuPropagator<double>::setEnergyList(list); // set host energy list

          //copy host energy list to gpu memory
          cudaMemcpy(d_energy_list.get(), this->energyList.data(), sizeof(double) * this->n_energies, H2D); CUERR;
        }

        void cudaprob3::AtmosCudaPropagatorSingle::setCosineList(const std::vector<double>& list) {
          AtmosCpuPropagator<double>::setCosineList(list); // set host cosine list
          //copy host cosine list to gpu memory
          cudaMemcpy(d_cosine_list.get(), this->cosineList.data(), sizeof(double) * this->n_cosines, H2D); CUERR;
        }

        void cudaprob3::AtmosCudaPropagatorSingle::setProductionHeightList(const std::vector<double>& list_prob, const std::vector<double>& list_bins) {
          AtmosCpuPropagator<double>::setProductionHeightList(list_prob, list_bins); //set host production height list

          cudaMemcpy(d_productionHeight_prob_list.get(), this->productionHeightList_prob.data(), sizeof(double)*Constants<double>::MaxProdHeightBins()*2*3*this->n_energies*this->n_cosines, H2D); CUERR;
          cudaMemcpy(d_productionHeight_bins_list.get(), this->productionHeightList_bins.data(), sizeof(double)*(Constants<double>::MaxProdHeightBins()+1), H2D); CUERR;
        }

        // calculate the probability of each cell
        void cudaprob3::AtmosCudaPropagatorSingle::calculateProbabilities(NeutrinoType type) {
          calculateAtmosProbabilitiesAsync(type);
          waitForCompletion();
        }

        void cudaprob3::AtmosCudaPropagatorSingle::setChemicalComposition(const std::vector<double>& list) {
          if (list.size() != this->yps.size()) {
            throw std::runtime_error("cudapropagator::setChemicalComposition. Size of input list not equal to expectation.");
          }

          for (std::uint64_t iyp=0;iyp<list.size();iyp++) {
            this->yps[iyp] = list[iyp];
          }

          int nDensityLayers = this->radii.size();

          cudaMemcpy(d_yps.get(), this->yps.data(), sizeof(double) * nDensityLayers, H2D); CUERR;
        }

        // get oscillation weight for specific cosine and energy
        double cudaprob3::AtmosCudaPropagatorSingle::getProbability(int index_cosine, int index_energy, ProbType t) {
          if(index_cosine >= this->n_cosines || index_energy >= this->n_energies)
            throw std::runtime_error("CudaPropagatorSingle::getProbability. Invalid indices");

          if(!resultsResideOnHost){
            getResultFromDevice();
            resultsResideOnHost = true;
          }

          const std::uint64_t index = std::uint64_t(index_cosine) * std::uint64_t(this->n_energies) + std::uint64_t(index_energy);
          const std::uint64_t offset = std::uint64_t(t) * std::uint64_t(this->n_energies) * std::uint64_t(this->n_cosines);

          return resultList.get()[index + offset];
        }

        // get oscillation weight for specific cosine and energy
        void cudaprob3::AtmosCudaPropagatorSingle::getProbabilityArr(double* probArr, ProbType t) {

          if(!resultsResideOnHost){
            getResultFromDevice();
            resultsResideOnHost = true;
          }

          double* resultList_Arr = resultList.get();
          const std::uint64_t offset = std::uint64_t(t) * std::uint64_t(this->n_energies) * std::uint64_t(this->n_cosines);

          std::uint64_t iter = 0;
          for (int index_energy=0;index_energy<this->n_energies;index_energy++) {
            for (int index_cosine=0;index_cosine<this->n_cosines;index_cosine++) {
              std::uint64_t index = std::uint64_t(index_cosine) * std::uint64_t(this->n_energies) + std::uint64_t(index_energy);
              probArr[iter] = resultList_Arr[index + offset];
              iter += 1;
            }
          }

        }

        void cudaprob3::AtmosCudaPropagatorSingle::setMaxlayers() {
          AtmosCpuPropagator<double>::setMaxlayers();

          cudaMemcpy(d_maxlayers.get(), this->maxlayers.data(), sizeof(int) * this->n_cosines, H2D); CUERR;
        }

        // launch the calculation kernel without waiting for its completion
        void cudaprob3::AtmosCudaPropagatorSingle::calculateAtmosProbabilitiesAsync(NeutrinoType type){
          if(!this->isInit)
            throw std::runtime_error("CudaPropagatorSingle::calculateProbabilities. Object has been moved from.");
          if(!this->isSetProductionHeight)
            throw std::runtime_error("CudaPropagatorSingle::calculateProbabilities. production height was not set");
          if(this->useProductionHeightAveraging && !this->isSetProductionHeightArray)
            throw std::runtime_error("CudaPropagatorSingle::calculateProbabilities. production height array was not set");

          resultsResideOnHost = false;
          cudaSetDevice(deviceId); CUERR;

          // set neutrino parameters for core physics functions for both host and device
          physics::setMixMatrix(this->Mix_U.data());
          physics::setMassDifferences(this->dm.data());

          dim3 block(64, 1, 1);

          //const unsigned blocks = SDIV(this->energyList.size() * this->cosineList.size(), block.x);
          const unsigned blocks = SDIV(this->energyList.size(), block.x) * this->cosineList.size();

          dim3 grid(blocks, 1, 1);

          physics::callCalculateAtmosKernelAsync(grid, block, stream,
              type,
              d_cosine_list.get(), 
              this->n_cosines,
              d_energy_list.get(), 
              this->n_energies,
              d_radii.get(), 
              d_as.get(), 
              d_bs.get(), 
              d_cs.get(), 
              d_rhos.get(), 
              d_yps.get(),
              d_maxlayers.get(),
              this->ProductionHeightinCentimeter,
              this->useProductionHeightAveraging,
              this->nProductionHeightBins,
              d_productionHeight_prob_list.get(),
              d_productionHeight_bins_list.get(),
              this->UsePolyDensity,
              d_result_list.get());

          CUERR;
        }

        // wait for calculateProbabilitiesAsync to finish
        void cudaprob3::AtmosCudaPropagatorSingle::waitForCompletion(){
          cudaSetDevice(deviceId); CUERR;
          cudaStreamSynchronize(stream); CUERR;
        }

        // copy results from device to host
        void cudaprob3::AtmosCudaPropagatorSingle::getResultFromDevice(){
          cudaSetDevice(deviceId); CUERR;
          cudaMemcpyAsync(resultList.get(), d_result_list.get(),
              sizeof(double) * std::uint64_t(9) * std::uint64_t(this->n_energies) * std::uint64_t(this->n_cosines),
              D2H, stream);  CUERR;
          cudaStreamSynchronize(stream);
        }

// Mutli bit commented out
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

#endif

#endif // #ifdef __NVCC__
