#include "cpupropagator.hpp" // include openmp propagator
#include "cudapropagator.cuh" // include cuda propagator
#include "hpc_helpers.cuh" // timer

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "TFile.h"
#include "TH2D.h"

using namespace cudaprob3; // namespace of the propagators

int main(int argc, char** argv){

  //using FLOAT_T = float;
  using FLOAT_T = double;

  TString InputFileName = "Oscillograms.root";
  TFile* InFile = new TFile(InputFileName);
  TH2D* TemplateHist = (TH2D*)InFile->Get("Osc/Fine/hSecondaryArray_0_0_0");

  int nNeutrinoTypes = 2;
  std::vector<NeutrinoType> NeutrinoTypes(nNeutrinoTypes);
  NeutrinoTypes[0] = Neutrino;
  NeutrinoTypes[1] = Antineutrino;

  std::vector<TString> NeutrinoTypes_Names(nNeutrinoTypes);
  NeutrinoTypes_Names[0] = "Neutrino";
  NeutrinoTypes_Names[1] = "Antineutrino";

  int nOscChannels = 6;
  std::vector<ProbType> OscChannels(nOscChannels);
  OscChannels[0] = e_e;
  OscChannels[1] = e_m;
  OscChannels[2] = e_t;
  OscChannels[3] = m_e;
  OscChannels[4] = m_m;
  OscChannels[5] = m_t;

  std::vector<TString> OscChannels_Names(nOscChannels);
  OscChannels_Names[0] = "e_e";
  OscChannels_Names[1] = "e_m";
  OscChannels_Names[2] = "e_t";
  OscChannels_Names[3] = "m_e";
  OscChannels_Names[4] = "m_m";
  OscChannels_Names[5] = "m_t";

  std::vector< std::vector<TH2D*> > vecHist(nNeutrinoTypes);
  for (int iNeutrinoTypes=0;iNeutrinoTypes<nNeutrinoTypes;iNeutrinoTypes++) {
    vecHist[iNeutrinoTypes].resize(nOscChannels);
    for (int iOscChannels=0;iOscChannels<nOscChannels;iOscChannels++) {
      vecHist[iNeutrinoTypes][iOscChannels] = (TH2D*)TemplateHist->Clone(Form("NuType_%i_OscChan_%i",iNeutrinoTypes,iOscChannels));
      vecHist[iNeutrinoTypes][iOscChannels]->SetTitle(NeutrinoTypes_Names[iNeutrinoTypes]+" "+OscChannels_Names[iOscChannels]);
    }
  }

  std::vector<FLOAT_T> CosineList;
  std::vector<FLOAT_T> EnergyList;

  std::cout << "Energy Values:" << std::endl;
  for (int xBin=1;xBin<=TemplateHist->GetNbinsX();xBin++) {
    EnergyList.push_back(TemplateHist->GetXaxis()->GetBinCenter(xBin));
    std::cout << "Index:" << std::setw(10) << xBin-1 << " | Val:" << std::setw(10) << EnergyList[xBin-1] << std::endl;
  }

  std::cout << "Cosine Values:" << std::endl;
  for (int yBin=1;yBin<=TemplateHist->GetNbinsY();yBin++) {
    CosineList.push_back(TemplateHist->GetYaxis()->GetBinCenter(yBin));
    std::cout << "Index:" << std::setw(10) << yBin-1 << " | Val:" << std::setw(10) << CosineList[yBin-1] << std::endl;
  }

  int nCosine = (int)CosineList.size();
  int nEnergy = (int)EnergyList.size();

  // Prob3++ probRoot.cc parameters in radians

  const FLOAT_T theta12 = 0.5872523687443223;
  const FLOAT_T theta13 = 0.4857872793291233;
  const FLOAT_T theta23 = 0.8134128187551903;
  const FLOAT_T dcp     = -1.601;
  
  const FLOAT_T dm12sq = 0.0000753;
  const FLOAT_T dm23sq = 0.002509;
	
  std::unique_ptr<Propagator<FLOAT_T>> propagator( new CpuPropagator<FLOAT_T>(nCosine, nEnergy, 4)); // cpu propagator with 4 threads
  
  // these 3 are only available if compiled with nvcc.
  
  //std::unique_ptr<Propagator<FLOAT_T>> propagator( new CudaPropagatorSingle<FLOAT_T>(0, nCosine, nEnergy)); // Single GPU propagator using GPU 0
  //std::unique_ptr<Propagator<FLOAT_T>> propagator( new CudaPropagator<FLOAT_T>(std::vector<int>{0}, n_cosines, n_energies)); // Multi GPU propagator which only uses GPU 0. Behaves identical to propagator above.
  //std::unique_ptr<Propagator<FLOAT_T>> propagator( new CudaPropagator<FLOAT_T>(std::vector<int>{0, 1, 2, 3}, n_cosines, n_energies)); // Multi GPU propagator which uses GPU 0 and GPU 1

  // set energy list
  propagator->setEnergyList(EnergyList);
  
  //set cosine list
  propagator->setCosineList(CosineList);

  // set mixing matrix. angles in radians
  propagator->setMNSMatrix(theta12, theta13, theta23, dcp);
  
  // set neutrino mass differences. unit: eV^2
  propagator->setNeutrinoMasses(dm12sq, dm23sq);

  // set density model
  propagator->setDensityFromFile("example/models/PREM_4layer.dat");

  // set neutrino production height in kilometers above earth
  propagator->setProductionHeight(25.0);
  
  for (int iNeutrinoTypes=0;iNeutrinoTypes<nNeutrinoTypes;iNeutrinoTypes++) {

    if (NeutrinoTypes[iNeutrinoTypes]==Antineutrino) {
      // DB, Haven't really thought about it, but prob3++ sets dcp->-dcp here: https://github.com/rogerwendell/Prob3plusplus/blob/fd189e232e96e2c5ebb2f7bd3a5406b288228e41/BargerPropagator.cc#L235
      // Copying that behaviour gives same behaviour as prob3++/probGPU
      propagator->setMNSMatrix(theta12, theta13, theta23, -dcp);
    } else {
      propagator->setMNSMatrix(theta12, theta13, theta23, dcp);
    }
    propagator->calculateProbabilities(NeutrinoTypes[iNeutrinoTypes]);

    for (int iOscChannels=0;iOscChannels<nOscChannels;iOscChannels++) {
      for(int CosineBin=0;CosineBin<nCosine;CosineBin++) {
	for(int EnergyBin=0;EnergyBin<nEnergy;EnergyBin++) {
	  //std::cout << EnergyBin << " " << CosineBin << " " << propagator->getProbability(EnergyBin,CosineBin,OscChannels[iOscChannels]) << std::endl;
	  vecHist[iNeutrinoTypes][iOscChannels]->SetBinContent(EnergyBin+1,CosineBin+1,propagator->getProbability(CosineBin,EnergyBin,OscChannels[iOscChannels]));
	}
      }
    }
  }

  TString OutputFileName = "CUDAProbOutput.root";
  TFile* OutFile = new TFile(OutputFileName,"RECREATE");

  for (int iNeutrinoTypes=0;iNeutrinoTypes<nNeutrinoTypes;iNeutrinoTypes++) {
    for (int iOscChannels=0;iOscChannels<nOscChannels;iOscChannels++) {
      vecHist[iNeutrinoTypes][iOscChannels]->Write();
    }
  }

  OutFile->Close();
  InFile->Close();
  std::cout << "Saved output to:" << OutputFileName << std::endl;

}
