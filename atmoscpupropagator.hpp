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

#ifndef CUDAPROB3_ATMOSCPUPROPAGATOR_HPP
#define CUDAPROB3_ATMOSCPUPROPAGATOR_HPP

#include "constants.hpp"
#include "propagator.hpp"
#include "cpupropagator.hpp"
#include "physics.hpp"

#include <omp.h>
#include <vector>


namespace cudaprob3linear{

    /// \class CpuPropagator
    /// \brief Multi-threaded CPU neutrino propagation. Derived from Propagator
    /// @param FLOAT_T The floating point type to use for calculations, i.e float, double
    template<class FLOAT_T>
    class AtmosCpuPropagator : public CpuPropagator<FLOAT_T>{
    public:
        /// \brief Constructor
        ///
        /// @param num_cosines Number cosine bins
        /// @param num_energies Number of energy bins
        /// @param threads Number of threads
        AtmosCpuPropagator(int num_cosines, int num_energies, int threads) : CpuPropagator<FLOAT_T>(num_cosines, num_energies, threads){

            resultList.resize(std::uint64_t(num_cosines) * std::uint64_t(num_energies) * std::uint64_t(9));
            omp_set_num_threads(threads);
        }

        /// \brief Copy constructor
        /// @param other
        AtmosCpuPropagator(const AtmosCpuPropagator& other) : CpuPropagator<FLOAT_T>(other){
            *this = other;
        }

        /// \brief Move constructor
        /// @param other
        AtmosCpuPropagator(AtmosCpuPropagator&& other) : CpuPropagator<FLOAT_T>(other){
            *this = std::move(other);
        }

        /// \brief Copy assignment operator
        /// @param other
        AtmosCpuPropagator& operator=(const AtmosCpuPropagator& other){
            CpuPropagator<FLOAT_T>::operator=(other);

            resultList = other.resultList;

            return *this;
        }

        /// \brief Move assignment operator
        /// @param other
        AtmosCpuPropagator& operator=(AtmosCpuPropagator&& other){
            CpuPropagator<FLOAT_T>::operator=(std::move(other));

            resultList = std::move(other.resultList);

            return *this;
        }

    public:

        // Returns the number of LAYER BOUNDARIES
        // i.e. number of layers is this number *MINUS ONE*
        virtual int getNlayerBoundaries() { return n_layers; };
       
 
        virtual void SetNumberOfProductionHeightBinsForAveraging(int nProductionHeightBins_) {
        if (nProductionHeightBins_ > Constants<FLOAT_T>::MaxProdHeightBins()) {
          std::cerr << "Invalid number of production height averages:" << nProductionHeightBins_ << std::endl;
          std::cerr << "Need to increase value of Constants<FLOAT_T>::MaxProdHeightBins() in $CUDAPROB3/constants.hpp" << std::endl;
          throw std::runtime_error("SetNumberOfProductionHeightBinsForAveraging : invalid number of production height bins");
        }

        this->nProductionHeightBins = nProductionHeightBins_;

        if (this->nProductionHeightBins >= 1) {
          this->useProductionHeightAveraging = false;
        }

        if (this->useProductionHeightAveraging == true) {
          std::cout << "Set " << this->nProductionHeightBins << " Production height bins" << std::endl;
        } else {
          std::cout << "Using fixed production height" << std::endl;
        }
      }


        /// \brief Set density information from arrays.
        /// \details radii_ and rhos_ must be same size. both radii_ and rhos_ must be sorted, in the same order.
        /// The density (g/cm^3) at a distance (km) from the center of the sphere between radii_[i], exclusive,
        /// and radii_[j], inclusive, i < j  is assumed to be rhos_[j]
        /// @param radii_ List of radii
        /// @param rhos_ List of densities
        /// @param yps_ List of chemical compositions
      virtual void setDensity(
          const std::vector<FLOAT_T>& radii_, 
          const std::vector<FLOAT_T>& rhos_, 
          const std::vector<FLOAT_T>& yps_){

          UsePolyDensity = false;

            if(rhos_.size() != radii_.size()){
                throw std::runtime_error("setDensity : rhos.size() != radii.size()");
            }

            if(rhos_.size() != yps_.size()){
	      throw std::runtime_error("setDensity : rhos.size() != yps.size()");
            }

            if(rhos_.size() == 0 || radii_.size() == 0 || yps_.size() == 0){
	      throw std::runtime_error("setDensity : vectors must not be empty");
            }

            bool needFlip = false;

            if(radii_.size() >= 2){
                int sign = (radii_[1] - radii_[0] > 0 ? 1 : -1);

                for(size_t i = 1; i < radii_.size(); i++){
                    if((radii_[i] - radii_[i-1]) * sign < 0)
                        throw std::runtime_error("radii order messed up");
                }

                if(sign == 1)
                    needFlip = true;
            }

            radii = radii_;
            rhos = rhos_;
	    yps = yps_;

            if(needFlip){
                std::reverse(radii.begin(), radii.end());
                std::reverse(rhos.begin(), rhos.end());
		std::reverse(yps.begin(), yps.end());
            }

            coslimit.clear();

            // first element of _Radii is largest radius
            for(size_t i=0; i < radii.size() ; i++ ) {
                // Using a cosine threshold
                FLOAT_T x = -1* sqrt( 1 - (radii[i] * radii[i] / ( Constants<FLOAT_T>::REarth()*Constants<FLOAT_T>::REarth())) );
                if ( i  == 0 ) x = 0;
                coslimit.push_back(x);
            }

            setMaxlayers();
        }

        /// \brief Set density information from arrays including polynomials for a non-constant density in each layer
        /// \details radii_ and rhos_ must be same size. both radii_ and rhos_ must be sorted, in the same order.
        /// The density (g/cm^3) at a distance (km) from the center of the sphere between radii_[i], exclusive,
        /// and radii_[j], inclusive, i < j  is assumed to be rhos_[j]
        /// @param radii_ List of radii
        /// @param a_ List of densities coefficient a
        /// @param b_ List of densities coefficient b
        /// @param c_ List of densities coefficient c
        /// @param yps_ List of chemical compositions
        virtual void setDensity(
          const std::vector<FLOAT_T>& radii_,
          const std::vector<FLOAT_T>& a_,
          const std::vector<FLOAT_T>& b_,
          const std::vector<FLOAT_T>& c_,
          const std::vector<FLOAT_T>& yps_) {

        UsePolyDensity = true;

        if(a_.size() != radii_.size()){
          throw std::runtime_error("setDensity : a.size() != radii.size()");
        }

        if(a_.size() != yps_.size()){
          throw std::runtime_error("setDensity : a.size() != yps.size()");
        }

        if(a_.size() == 0 || b_.size() == 0 || c_.size() == 0 || radii_.size() == 0 || yps_.size() == 0){
          throw std::runtime_error("setDensity : vectors must not be empty");
        }

        bool needFlip = false;

        if(radii_.size() >= 2){
          int sign = (radii_[1] - radii_[0] > 0 ? 1 : -1);

          for(size_t i = 1; i < radii_.size(); i++){
            if((radii_[i] - radii_[i-1]) * sign < 0)
              throw std::runtime_error("radii order messed up");
          }

          if(sign == 1) needFlip = true;
        }


        //Copy over the content (probably unnecessary...)
        radii = radii_;
        as = a_;
        bs = b_;
        cs = c_;
        yps = yps_;
        
        if(needFlip){
          std::reverse(radii.begin(), radii.end());
          std::reverse(yps.begin(), yps.end());
          std::reverse(as.begin(), as.end());
          std::reverse(bs.begin(), bs.end());
          std::reverse(cs.begin(), cs.end());
        }

        coslimit.clear();

        //first element of _Radii is largest radius
        for(size_t i=0; i < radii.size() ; i++ ) {
          // Using a cosine threshold
          FLOAT_T x = -1* sqrt( 1 - (radii[i] * radii[i] / ( Constants<FLOAT_T>::REarth()*Constants<FLOAT_T>::REarth())) );
          if ( i  == 0 ) x = 0;
          coslimit.push_back(x);
        }
        
        setMaxlayers();
     }                                                                                                                                                                       
        
      /// \brief Set density information from file
      /// \details File must contain two columns where the first column contains the radius (km)
      /// and the second column contains the density (g/cm³).
      /// The first row must have the radius 0. The last row must have to radius of the sphere
      ///
      /// @param filename File with density information
      virtual void setDensityFromFile(const std::string& filename){
        std::ifstream file(filename);
        if(!file)
          throw std::runtime_error("could not open density file " + filename);

        std::vector<FLOAT_T> radii_temp;
        std::vector<FLOAT_T> rhos_temp;
        std::vector<FLOAT_T> yps_temp;

        // First check if the file contains rho or polynomial coefficient
        // reading the first line should suffice

        std::string line;
        int nentries_old = 0;
        if (file.is_open()) {
          while (std::getline(file, line)) {
            // Allow for comments or empty lines
            if (line[0] == '#' || line.empty()) continue;
            // Check how many entries we have per line
            int nentries = 0;
            while (line.find_first_of(" ") != std::string::npos) {
              int newpos = line.find_first_of(" ");
              std::string substring = line.substr(0, newpos);
              // Check repeated spaces
              while (line[newpos] == line[newpos+1]) newpos++;
              line = line.substr(newpos+1, line.size());
              nentries++;
            }
            nentries++;
            if (nentries_old == 0) nentries_old = nentries;
            if (nentries != nentries_old) std::cout << "Inconsitent number of entries" << std::endl;
          }
        }

        std::cout << "Found " << nentries_old << " entries in file " << filename << std::endl;
        // Reset the file reader
        file.clear();
        file.seekg(0);

        // If the file is formatted as radius, density, electron fraction
        if (nentries_old == 3) {
          FLOAT_T r;
          FLOAT_T d;
          FLOAT_T yp;
          while (file >> r >> d >> yp){
            radii_temp.push_back(r);
            rhos_temp.push_back(d);
            yps_temp.push_back(yp);
          }

          setDensity(radii_temp, rhos_temp, yps_temp);
        } 
        else if (nentries_old == 5) {

          // Coefficients of density
          std::vector<FLOAT_T> a_temp;
          std::vector<FLOAT_T> b_temp;
          std::vector<FLOAT_T> c_temp;

          if (file.is_open()) {
            while (std::getline(file, line)) {
              std::vector<std::string> entries;
              // Allow for comments or empty lines
              if (line[0] == '#' || line.empty()) continue;
              // Check how many entries we have per line
              while (line.find_first_of(" ") != std::string::npos) {
                int newpos = line.find_first_of(" ");
                std::string substring = line.substr(0, newpos);
                entries.push_back(substring);
                while (line[newpos] == line[newpos+1]) newpos++;
                line = line.substr(newpos+1, line.size());
              }
              entries.push_back(line);

              // Now push back into our main vectors
              radii_temp.push_back(std::atof(entries[0].c_str()));
              a_temp.push_back(std::atof(entries[1].c_str()));
              b_temp.push_back(std::atof(entries[2].c_str()));
              c_temp.push_back(std::atof(entries[3].c_str()));
              yps_temp.push_back(std::atof(entries[4].c_str()));
            }
          }
          //for (int i = 0; i < nentries_old; ++i) {
            //std::cout << radii_temp[i] << " " << a_temp[i] << " " << b_temp[i] << " " << c_temp[i] << " " << yps_temp[i] << std::endl;
          //}

          setDensity(radii_temp, a_temp, b_temp, c_temp, yps_temp);

        } else {
          std::cout << "Unsupported earty model in " << filename << std::endl;
          std::cout << "  Number of entries per line: " << nentries_old << std::endl;
          throw;
        }
        n_layers = radii.size();
      }
      
        // Currently a dummy function
        virtual void setDensity( FLOAT_T rho ) {
          (void) rho;
	      std::cout << "DUMMY FUNCTION: ATMOS class uses setDensity( \n" ;
          std::cout << "const std::vector<FLOAT_T>& radii_, \n " ;
          std::cout << "const std::vector<FLOAT_T>& a_, \n " ;
          std::cout << "const std::vector<FLOAT_T>& b_, \n " ;
          std::cout << "const std::vector<FLOAT_T>& c_, \n " ;
          std::cout << "const std::vector<FLOAT_T>& yps_) \n " ;
		  std::cout << "or \n " ;
          std::cout << "setDensityFromFile(const std::string& filename) " << std::endl;
        }


        // Currently a dummy function
        virtual void setPathLength( FLOAT_T path_length ) {
          (void) path_length;
		  std::cout << "DUMMY FUNCTION - ATMOS class calculates PATH LENGTH" << std::endl; 
        }

      /// \brief Set cosine bins. Cosines are given in radians
      /// @param list Cosine list
      virtual void setCosineList(const std::vector<FLOAT_T>& list){
        if(list.size() != size_t(this->n_cosines)){
          throw std::runtime_error("Propagator::setCosineList. Propagator was not created for this number of cosine nodes");}

        cosineList = list;

        if(isSetProductionHeight){
          setProductionHeight(ProductionHeightinCentimeter / 100000.0);
        }

        setMaxlayers();

        isSetCosine = true; 
      }

      /// \brief Set production height in km of neutrinos
      /// \details Adds a layer of length heightKM with zero density to the density model
      /// @param heightKM Set neutrino production height
      virtual void setProductionHeight(FLOAT_T heightKM){
        if(!isSetCosine)
          throw std::runtime_error("must set cosine list before production height");

        ProductionHeightinCentimeter = heightKM * 100000.0;

        isSetProductionHeight = true;
      }

      virtual void setProductionHeightList(const std::vector<FLOAT_T>& list_prob, const std::vector<FLOAT_T>& list_bins) {
        if (!this->useProductionHeightAveraging) {
          throw std::runtime_error("Propagator::setProductionHeightList. Trying to set Production Height information but propagator is not expecting to use it");
        }

        if (int(list_prob.size()) != this->nProductionHeightBins*2*3*this->n_energies*this->n_cosines) {
          throw std::runtime_error("Propagator::setProductionHeightList. Prob array is not the expected size");
        }

        if (int(list_bins.size())-1 != this->nProductionHeightBins) {
          throw std::runtime_error("Propagator::setProductionHeightList. ProductionHeightBins array is not expected size");
        }

        int MaxSize = Constants<FLOAT_T>::MaxProdHeightBins()*2*3*this->n_energies*this->n_cosines;
        productionHeightList_prob = std::vector<FLOAT_T>(MaxSize);
        for (int i=0;i<MaxSize;i++) {
          productionHeightList_prob[i] = 0.;
  }
        for (unsigned int i=0;i<list_prob.size();i++) {
          productionHeightList_prob[i] = list_prob[i];
        }

        productionHeightList_bins = std::vector<FLOAT_T>(Constants<FLOAT_T>::MaxProdHeightBins()+1);
        for (unsigned int i=0;i<Constants<FLOAT_T>::MaxProdHeightBins();i++) {
          productionHeightList_bins[i] = 0.;
        }

        for (unsigned int i=0;i<list_bins.size();i++) {
          productionHeightList_bins[i] = list_bins[i];
        }

        isSetProductionHeightArray = true;
      }

      // for each cosine bin, determine the number of layers which will be crossed by the neutrino path
      // the atmospheric layers is excluded
      virtual void setMaxlayers(){
        for(int index_cosine = 0; index_cosine < this->n_cosines; index_cosine++){
          FLOAT_T c = cosineList[index_cosine];
          const int maxLayer = std::count_if(coslimit.begin(), coslimit.end(), [c](FLOAT_T limit){ return c < limit;});

          if (maxLayer > Constants<FLOAT_T>::MaxNLayers()) {
            std::cerr << "Invalid number of maxLayer:" << maxLayer << std::endl;
            std::cerr << "Need to increase value of Constants<FLOAT_T>::MaxNLayers() in $CUDAPROB3/constants.hpp" << std::endl;
            throw std::runtime_error("setMaxlayers : invalid number of maxLayer");
          }

          this->maxlayers[index_cosine] = maxLayer;
        }
      }


        virtual void calculateProbabilities(NeutrinoType type) override{

            if(!this->isInit)
                throw std::runtime_error("CpuPropagator::calculateProbabilities. Object has been moved from.");
            if(!this->isSetProductionHeight)
                throw std::runtime_error("CpuPropagator::calculateProbabilities. production height was not set");
            if(this->useProductionHeightAveraging && !this->isSetProductionHeightArray)
                throw std::runtime_error("CpuPropagator::calculateProbabilities. production height array was not set, but been requested to use production height averaging");

            // set neutrino parameters for core physics functions
            physics::setMixMatrix_host(this->Mix_U.data());
            physics::setMassDifferences_host(this->dm.data());

            physics::calculate_atmos(type, 
                this->cosineList.data(), 
                this->cosineList.size(),
                this->energyList.data(), 
                this->energyList.size(), 
                this->radii.data(), 
                this->as.data(), 
                this->bs.data(), 
                this->cs.data(), 
                this->rhos.data(), 
                this->yps.data(), 
                this->maxlayers.data(), 
                this->ProductionHeightinCentimeter,
                this->useProductionHeightAveraging,
                this->nProductionHeightBins,
                this->productionHeightList_prob.data(), 
                this->productionHeightList_bins.data(), 
                this->UsePolyDensity, // Are we using constant density or polynomial?
                resultList.data());
        }

        virtual void setChemicalComposition(const std::vector<FLOAT_T>& list) override{
          if (list.size() != this->yps.size()) {
            throw std::runtime_error("CpuPropagator::setChemicalComposition. Size of input list not equal to expectation.");
          }

          for (std::uint64_t iyp=0;iyp<list.size();iyp++) {
            this->yps[iyp] = list[iyp];
          }

        }

       virtual  FLOAT_T getProbability(int index_cosine, int index_energy, ProbType t) {
          if(index_cosine >= this->n_cosines || index_energy >= this->n_energies) {
            throw std::runtime_error("CpuPropagator::getProbability. Invalid indices");
          }

          std::uint64_t index = std::uint64_t(index_cosine) * std::uint64_t(this->n_energies) * std::uint64_t(9)
            + std::uint64_t(index_energy) * std::uint64_t(9);
          return resultList[index + int(t)];
        }

        virtual void getProbabilityArr(FLOAT_T* probArr, ProbType t) {

          std::uint64_t iter = 0;
          for (int index_energy=0;index_energy<this->n_energies;index_energy++) {
            for (int index_cosine=0;index_cosine<this->n_cosines;index_cosine++) {
              std::uint64_t index = std::uint64_t(index_cosine) * std::uint64_t(this->n_energies) * std::uint64_t(9) + std::uint64_t(index_energy) * std::uint64_t(9);
              probArr[iter] = resultList[index + int(t)];
              iter += 1;
            }
          }

        }

    protected:
      
      std::vector<FLOAT_T> cosineList;

      std::vector<FLOAT_T> productionHeightList_prob;
      std::vector<FLOAT_T> productionHeightList_bins;
        
      std::vector<FLOAT_T> radii;
      std::vector<FLOAT_T> rhos;
      std::vector<FLOAT_T> as;
      std::vector<FLOAT_T> bs;
      std::vector<FLOAT_T> cs;
      std::vector<FLOAT_T> yps;
      std::vector<FLOAT_T> coslimit;

      FLOAT_T ProductionHeightinCentimeter;
      
      bool useProductionHeightAveraging = false;
      int nProductionHeightBins = 0;

      bool isSetProductionHeightArray = false;
      bool isSetProductionHeight = false;
      bool isSetCosine = false;
      bool isInit = true;

      int n_layers;


      // Use polynomial density for density averaging each track?
      bool UsePolyDensity;
    private:
        std::vector<FLOAT_T> resultList;
    };



}

#endif
