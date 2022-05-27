enable_language(CUDA)

cmessage("I have enabled CUDA!!!")

#if(NOT DEFINED CUDA_SAMPLES)
 # cmessage(FATAL_ERROR "When using CUDA, CUDA_SAMPLES must be defined to point to the CUDAToolkit samples directory (should contain common/helper_functions.h).")
#endif()

find_package(CUDAToolkit)

#add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-I${CUDA_SAMPLES}/common/inc>")

EXECUTE_PROCESS( COMMAND uname -m OUTPUT_VARIABLE OS_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT "x86_64 " STREQUAL "${OS_ARCH} ")
	cmessage(FATAL_ERROR "This build currently only support x86_64 target arches, determined the arch to be: ${OS_ARCH}")
endif()

EXECUTE_PROCESS( COMMAND ${CMAKE_SOURCE_DIR}/cmake/cudaver.sh --major OUTPUT_VARIABLE CUDA_MAJOR_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

cmessage(STATUS "CUDA_MAJOR_VERSION: ${CUDAToolkit_VERSION}")

add_compile_definitions(CUDA)

add_compile_options(
  	"$<$<COMPILE_LANGUAGE:CUDA>:-g;-O2;-lineinfo>"
  	"$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp;-Xcompiler=-Wall>"
  	"$<$<COMPILE_LANGUAGE:CUDA>:-I>"
  	)
