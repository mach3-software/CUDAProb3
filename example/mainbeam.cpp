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

#include <beamcpupropagator.hpp> // include openmp propagator
#include <beamcudapropagator.cuh> // include openmp propagator
//#include <atmoscpupropagator.hpp> // include openmp propagator
#include <hpc_helpers.cuh> // timer


#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace cudaprob3; // namespace of the propagators

template<class T>
std::vector<T> linspace(T Emin,T Emax,unsigned int div){
  if(div==0)
    throw std::length_error("div == 0");

  std::vector<T> linpoints(div, 0.0);

  T step_lin = (Emax - Emin)/T(div-1);

  T EE = Emin;

  for(unsigned int i=0; i<div-1; i++, EE+=step_lin)
    linpoints[i] = EE;

  linpoints[div-1] = Emax;

  return linpoints;
}

template<class T>
std::vector<T> logspace(T Emin,T Emax,unsigned int div){
  if(div==0)
    throw std::length_error("div == 0");
  std::vector<T> logpoints(div, 0.0);

  T Emin_log,Emax_log;
  Emin_log = log(Emin);
  Emax_log = log(Emax);

  T step_log = (Emax_log - Emin_log)/T(div-1);

  logpoints[0]=Emin;
  T EE = Emin_log+step_log;
for(unsigned int i=1; i<div-1; i++, EE+=step_log)
    logpoints[i] = exp(EE);
  logpoints[div-1]=Emax;
  return logpoints;
}


int main(int argc, char** argv){

  using FLOAT_T = double;

  //// Binning
  int n_energies = 1e6;

  //FLOAT_T oscArr[n_energies+19];
  //if(argc > 3)
  //	n_threads = std::atoi(argv[3]);

  //std::vector<FLOAT_T> energyList = linspace((FLOAT_T)1.e-1, (FLOAT_T)50.e1, n_energies);
  std::vector<FLOAT_T> energyList(n_energies, 3.47279);

  // Prob3++ probRoot.cc parameters in radians
  const FLOAT_T theta12 = asin(sqrt(0.3097));
  const FLOAT_T theta13 = asin(sqrt(0.02241));
  const FLOAT_T theta23 = asin(sqrt(0.580));
  const FLOAT_T dcp     = 0;
  const FLOAT_T dm12sq = 7.39e-5;
  const FLOAT_T dm23sq = 2.4511e-3;

  double rho = 2.848;

//#ifndef USE_CPU
  //int n_threads = 1;
  //BeamCpuPropagator<FLOAT_T> *propagator; // cpu propagator with 4 threads
  //propagator = new BeamCpuPropagator<FLOAT_T>(n_energies, n_threads); // cpu propagator with 4 threads
  
  BeamCudaPropagatorSingle<FLOAT_T> *propagator; // cpu propagator with 4 threads
  propagator = new BeamCudaPropagatorSingle<FLOAT_T>(0, n_energies); // cpu propagator with 4 threads
//#else

  // these 3 are only available if compiled with nvcc.

  //propagator = new BeamCudaPropagatorSingle<FLOAT_T>(n_energies); // cpu propagator with 4 threads
  //std::unique_ptr<Propagator<FLOAT_T>> propagator( new BeamCudaPropagatorSingle<FLOAT_T>(n_energies)); // Single GPU propagator using GPU 0
//#endif
  //std::unique_ptr<Propagator<FLOAT_T>> propagator( new CudaPropagator<FLOAT_T>(std::vector<int>{0}, n_cosines, n_energies)); // Multi GPU propagator which only uses GPU 0. Behaves identical to propagator above.
  //std::unique_ptr<Propagator<FLOAT_T>> propagator( new CudaPropagator<FLOAT_T>(std::vector<int>{0, 1, 2, 3}, n_cosines, n_energies)); // Multi GPU propagator which uses GPU 0 and GPU 1


  // set energy list
  propagator->setEnergyList(energyList);

  // set mixing matrix. angles in radians
  propagator->setMNSMatrix(theta12, theta13, theta23, dcp);

  // set neutrino mass differences. unit: eV^2
  propagator->setNeutrinoMasses(dm12sq, dm23sq);

  // set density model
  propagator->setDensity(rho);

  // set Path length 
  propagator->setPathLength(1284.9);
 
  //std::vector<FLOAT_T> prob;
  //std::vector<FLOAT_T> height;

  auto start = std::chrono::high_resolution_clock::now();
  propagator->calculateProbabilities(cudaprob3::Neutrino);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration  = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "Time taken setup: " << duration.count() << " ms " << std::endl;

  //first result access after calculation triggers data transfer
  //propagator->getProbability(0, ProbType::m_m);

  // write output to files

  //std::ofstream outfile10("out_cudaboth_me.txt");
  //std::ofstream outfile11("out_cudaboth_mm.txt");
  //std::ofstream outfile12("out_cudaboth_mt.txt");


  //auto start2 = std::chrono::high_resolution_clock::now();
  //for(int i = 0; i < 10; i++) {
    //propagator->calculateProbabilities(cudaprob3::Neutrino);

    // ProbType::x_y is probability of transition x -> y
    //outfile10 << std::setprecision(100) << propagator->getProbability(i, ProbType::m_e) << " ";
    //outfile11 << std::setprecision(100) << propagator->getProbability(i, ProbType::m_m) << " ";
    //std::cout << std::setprecision(100) << propagator->getProbability(i, ProbType::m_m) << std::endl;
  //}

  //auto stop2 = std::chrono::high_resolution_clock::now();
  //auto duration2  = std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - start2);
  //std::cout << "Time taken calc: " << duration2.count() << " ms " << std::endl;
  //outfile10 << '\n';
  //outfile11 << '\n';
  //outfile12 << '\n';

  //outfile10.flush();
  //outfile11.flush();
  //outfile12.flush();



}
