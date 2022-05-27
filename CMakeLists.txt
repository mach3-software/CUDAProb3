add_library(beamCUDAProb3 SHARED beamcudapropagator.cu)
add_library(atmosCUDAProb3 SHARED atmoscudapropagator.cu)
#add_library(CUDAProb3 SHARED beamcudapropagator.cu atmoscudapropagator.cu)


set_target_properties(beamCUDAProb3 atmosCUDAProb3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA)
set_property(TARGET beamCUDAProb3 atmosCUDAProb3 PROPERTY CUDA_ARCHITECTURES 35 52 60 61 70 75 80 86)
#set_target_properties(CUDAProb3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CUDA)
#set_property(TARGET CUDAProb3 PROPERTY CUDA_ARCHITECTURES 35 52 60 61 70 75 80 86)

target_link_libraries(beamCUDAProb3)
target_link_libraries(atmosCUDAProb3)
#target_link_libraries(CUDAProb3)

install(TARGETS beamCUDAProb3 atmosCUDAProb3 DESTINATION lib)
