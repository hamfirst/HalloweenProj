# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)

add_executable(TestVid TestVid.cpp)
target_link_libraries(TestVid LandmarkDetector)
target_link_libraries(TestVid FaceAnalyser)
target_link_libraries(TestVid GazeAnalyser)
target_link_libraries(TestVid Utilities)

install (TARGETS TestVid DESTINATION bin)
