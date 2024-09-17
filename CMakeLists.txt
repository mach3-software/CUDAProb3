cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

#Use the compilers found in the path
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)
find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

if(NOT DEFINED GPU_ON)
  message(FATAL_ERROR "GPU_ON is not defined")
endif()

if(${GPU_ON})
  project(CUDAProb3 LANGUAGES CXX C CUDA)
  message(STATUS "#### CUDAProb3 with CPU and GPU ######")
else()
  project(CUDAProb3 LANGUAGES CXX C)
  message(STATUS "#### CUDAProb3 with CPU ONLY ######")
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


if(${GPU_ON})

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
  
  #KS: Allow user to define CMAKE_CUDA_ARCHITECTURES
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    #KS: See this for more info https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
  if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.23 )
    set(CMAKE_CUDA_ARCHITECTURES all )
    #KS: Consider using native, requires cmake 3.24... will be terrible for containers but should results in more optimised code
    #set(CMAKE_CUDA_ARCHITECTURES native )
  else()
    #KS: Apparently with newer cmake and GPU
    set(CMAKE_CUDA_ARCHITECTURES 35 52 60 61 70 75 80 86)
  endif()
  #KS: Bit hacky but to properly pass cuda flags we need
  string(REPLACE ";" " " CMAKE_CUDA_ARCHITECTURES_STRING "${CMAKE_CUDA_ARCHITECTURES}")
  else()
  #KS this may look hacky however CPM isn't build for passing stuff like this. If CMAKE_CUDA_ARCHITECTURES is passed CPM it will be  string not list. Thus we convert it to list
  set(CMAKE_CUDA_ARCHITECTURES_STRING ${CMAKE_CUDA_ARCHITECTURES})
  string(REPLACE " " ";" CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  endif()

  set_property(TARGET CUDAProb3Beam PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  set_property(TARGET CUDAProb3Atmos PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

  add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:GPU_ON>)

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

else()

  add_library(CUDAProb3Beam INTERFACE)
  add_library(CUDAProb3Atmos INTERFACE)

  target_include_directories(CUDAProb3Beam INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                  $<INSTALL_INTERFACE:include>)
  target_include_directories(CUDAProb3Atmos INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
                  $<INSTALL_INTERFACE:include>)

endif()
