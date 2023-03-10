cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

#Use the compilers found in the path
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

if(${CPU_ONLY})
  project(CUDAProb3 LANGUAGES CXX C)
else()
  project(CUDAProb3 LANGUAGES CXX C CUDA)
endif()

#Changes default install path to be a subdirectory of the build dir.
#Can set build dir at configure time with -DCMAKE_INSTALL_PREFIX=/install/path
if(CMAKE_INSTALL_PREFIX STREQUAL "" OR CMAKE_INSTALL_PREFIX STREQUAL
  "/usr/local")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
elseif(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/${CMAKE_SYSTEM_NAME}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
elseif(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()


if(${CPU_ONLY})

  add_library(CUDAProb3Beam INTERFACE)
  add_library(CUDAProb3Atmos INTERFACE)

  target_include_directories(CUDAProb3Beam INTERFACE ${PROJECT_SOURCE_DIR})
  target_include_directories(CUDAProb3Atmos INTERFACE ${PROJECT_SOURCE_DIR})

else()


  SET(HEADERS_BEAM beamcudapropagator.cuh)
  SET(HEADERS_ATMOS atmoscudapropagator.cuh)
  
   
  SET(SOURCE_BEAM beamcudapropagator.cu)
  SET(SOURCE_ATMOS atmoscudapropagator.cu)
  
  
  add_library(CUDAProb3Beam SHARED ${SOURCE_BEAM})
  add_library(CUDAProb3Atmos SHARED ${SOURCE_ATMOS})
  
  set_target_properties(CUDAProb3Beam PROPERTIES 
  	PUBLIC_HEADER "${HEADERS_BEAM}"
  	EXPORT_NAME CUDAProb3Beam
          CUDA_SEPARABLE_COMPILATION ON 
          LINKER_LANGUAGE CUDA)
  
  set_target_properties(CUDAProb3Atmos PROPERTIES 
  	PUBLIC_HEADER "${HEADERS_ATMOS}"
  	EXPORT_NAME CUDAProb3Atmos
          CUDA_SEPARABLE_COMPILATION ON 
          LINKER_LANGUAGE CUDA)
  
  set_property(TARGET CUDAProb3Beam PROPERTY CUDA_ARCHITECTURES 35 52 60 61 70 75)
  set_property(TARGET CUDAProb3Atmos PROPERTY CUDA_ARCHITECTURES 35 52 60 61 70 75)
  
  target_include_directories(
    CUDAProb3Beam PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                         $<INSTALL_INTERFACE:include>
  )
  
  target_include_directories(
    CUDAProb3Atmos PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                         $<INSTALL_INTERFACE:include>
  )
  
  install(TARGETS CUDAProb3Beam CUDAProb3Atmos 
  		EXPORT CUDAProb3-target
  		LIBRARY DESTINATION lib/
  		PUBLIC_HEADER DESTINATION include/)
  
  install(EXPORT CUDAProb3-target
    FILE CUDAProb3Targets.cmake
    NAMESPACE CUDAProb3::
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/CUDAProb3
  )

endif()

#include(CMakePackageConfigHelpers)
#configure_package_config_file(
#  CUDAProb3Config.cmake.in ${CMAKE_BINARY_DIR}/Prob3plusplusConfig.cmake
#  INSTALL_DESTINATION 
#  	/this/is/ignored/for/some/reason/thanks/kitware
#  NO_SET_AND_CHECK_MACRO
#  NO_CHECK_REQUIRED_COMPONENTS_MACRO)
#
#install(FILES 
#		${CMAKE_BINARY_DIR}/CUDAProb3Config.cmake 
#	DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/CUDAProb3)
#
#write_basic_package_version_file(${CMAKE_INSTALL_PREFIX}/lib/cmake/CUDAProb3/CUDAProb3ConfigVersion.cmake
  #VERSION ${CUDAProb3_VERSION}
 # COMPATIBILITY AnyNewerVersion)
