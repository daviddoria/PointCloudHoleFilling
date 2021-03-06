cmake_minimum_required(VERSION 2.6)

PROJECT(PointCloudHoleFilling)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(ROOT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Tell CMake to also look in the source directory to find some .cmake files
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

# Where to copy executables when 'make install' is run
SET(INSTALL_DIR ${CMAKE_INSTALL_PREFIX} )
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) #fix the dynamic linking error (RPATH) that would occur without this

# Enable c++11 and openmp.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++11 -fopenmp")

# Setup include directories for Qt
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# VTK
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})
set(PointCloudHoleFilling_libraries ${PointCloudHoleFilling_libraries} ${VTK_LIBRARIES})

if( "${VTK_MAJOR_VERSION}" LESS 6 )
  MESSAGE(FATAL_ERROR "You must build this code with VTK >= 6.0!")
endif( "${VTK_MAJOR_VERSION}" LESS 6 )

# ITK
FIND_PACKAGE(ITK REQUIRED ITKCommon ITKIOImageBase ITKIOPNG ITKIOMeta
             ITKImageIntensity ITKImageFeature ITKMathematicalMorphology ITKBinaryMathematicalMorphology ITKDistanceMap)
INCLUDE(${ITK_USE_FILE})
set(PointCloudHoleFilling_libraries ${PointCloudHoleFilling_libraries} ${ITK_LIBRARIES})

# Boost
FIND_PACKAGE(Boost 1.51 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Check for Qt4. If it is available, build the PatchBasedInpainting library using it so that SelfPatchCompare can use QtConcurrent.
# FIND_PACKAGE(Qt4)
# if(QT4_FOUND)
#   message ( "QT4 found successfully." )
#   INCLUDE(${QT_USE_FILE})
#   list(APPEND compileflags "USE_QT_PARALLEL")
# else ( QT4_FOUND )
#   message ( "QT4 NOT found successfully." )
# endif ( QT4_FOUND )

FIND_PACKAGE(Qt4)
INCLUDE(${QT_USE_FILE})
list(APPEND compileflags "USE_QT_PARALLEL")

#### Eigen ####

# Set the CMake variable to the value of the environment variable.
# If the build is absolutely fresh, this will be the case if the CMake variable EIGEN3_INCLUDE_DIR is not set.
if(NOT EIGEN3_INCLUDE_DIR)
  set(EIGEN3_INCLUDE_DIR $ENV{EIGEN3_INCLUDE_DIR})
  message("Set EIGEN3_INCLUDE_DIR to ${EIGEN3_INCLUDE_DIR} from environment variable of the same name.")
else()
  message("EIGEN3_INCLUDE_DIR is ${EIGEN3_INCLUDE_DIR}")
endif()

if(NOT EIGEN3_FOUND)
  FIND_PACKAGE(Eigen3 REQUIRED) #requires FindEigen3.cmake to be in the source directory
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Submodules
# Only include PatchBasedInpainting if it is not already available.
get_property(FoundPatchBasedInpainting GLOBAL PROPERTY PatchBasedInpaintingIncludeDirs SET)
if(NOT FoundPatchBasedInpainting)
  add_subdirectory(PatchBasedInpainting)
endif()

get_property(PatchBasedInpaintingIncludeDirs GLOBAL PROPERTY PatchBasedInpaintingIncludeDirs)
set(PointCloudHoleFilling_includes ${PointCloudHoleFilling_includes} ${PatchBasedInpaintingIncludeDirs})
get_property(PatchBasedInpaintingLibs GLOBAL PROPERTY PatchBasedInpaintingLibs)
set(PointCloudHoleFilling_libraries ${PointCloudHoleFilling_libraries} ${PatchBasedInpaintingLibs})

# Only include PoissonEditing if it is not already available.
get_property(FoundPoissonEditing GLOBAL PROPERTY PoissonEditingIncludeDirs SET)
if(NOT FoundPoissonEditing)
  add_subdirectory(PoissonEditing)
endif()

get_property(PoissonEditingIncludeDirs GLOBAL PROPERTY PoissonEditingIncludeDirs)
set(PointCloudHoleFilling_includes ${PointCloudHoleFilling_includes} ${PoissonEditingIncludeDirs})
get_property(PoissonEditingLibs GLOBAL PROPERTY PoissonEditingLibs)
set(PointCloudHoleFilling_libraries ${PointCloudHoleFilling_libraries} ${PoissonEditingLibs})

# Only include SmallHoleFiller if it is not already available.
get_property(FoundSmallHoleFiller GLOBAL PROPERTY SmallHoleFillerIncludeDirs SET)
if(NOT FoundSmallHoleFiller)
  add_subdirectory(SmallHoleFiller)
endif()

get_property(SmallHoleFillerIncludeDirs GLOBAL PROPERTY SmallHoleFillerIncludeDirs)
set(PointCloudHoleFilling_includes ${PointCloudHoleFilling_includes} ${SmallHoleFillerIncludeDirs})
get_property(SmallHoleFillerLibs GLOBAL PROPERTY SmallHoleFillerLibs)
set(PointCloudHoleFilling_libraries ${PointCloudHoleFilling_libraries} ${SmallHoleFillerLibs})

# Only include PTXTools if it is not already available.
get_property(FoundPTXTools GLOBAL PROPERTY PTXToolsIncludeDirs SET)
if(NOT FoundPTXTools)
  add_subdirectory(PTXTools)
endif()

get_property(PTXToolsIncludeDirs GLOBAL PROPERTY PTXToolsIncludeDirs)
set(PointCloudHoleFilling_includes ${PointCloudHoleFilling_includes} ${PTXToolsIncludeDirs})
get_property(PTXToolsLibs GLOBAL PROPERTY PTXToolsLibs)
set(PointCloudHoleFilling_libraries ${PointCloudHoleFilling_libraries} ${PTXToolsLibs})

# Check for submodules
if( (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PatchBasedInpainting/.git") OR
(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PoissonEditing/.git") OR
(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PTXTools/.git") )
  message( SEND_ERROR "At least one of the git submodules is not available.
                       Please run git submodule update --init --recursive")
endif()

# Give the compiler all required include directories
include_directories(${PointCloudHoleFilling_includes})

# This is the main executable to fill large holes in point clouds

option(PointCloudHoleFilling_PointCloudHoleFilling "Build the best PointCloudHoleFilling executable?" ON)
if(PointCloudHoleFilling_PointCloudHoleFilling)
    ADD_EXECUTABLE(PointCloudHoleFilling PointCloudHoleFilling.cpp)
    TARGET_LINK_LIBRARIES(PointCloudHoleFilling ${PointCloudHoleFilling_libraries} gomp)
    INSTALL( TARGETS PointCloudHoleFilling RUNTIME DESTINATION ${INSTALL_DIR} )
endif()

# This executable is for comparison purposes - it shows what happens when a point cloud hole is filled using a traditional inpainting approach
option(PointCloudHoleFilling_PointCloudHoleFillingNaive "Build the best PointCloudHoleFilling executable?" ON)
if(PointCloudHoleFilling_PointCloudHoleFillingNaive)
    ADD_EXECUTABLE(PointCloudHoleFillingNaive PointCloudHoleFillingNaive.cpp)
    TARGET_LINK_LIBRARIES(PointCloudHoleFillingNaive ${PointCloudHoleFilling_libraries} gomp)
    INSTALL( TARGETS PointCloudHoleFillingNaive RUNTIME DESTINATION ${INSTALL_DIR} )
endif()

# This executable reconstructs a point cloud from a filled RGBDxDy image
option(PointCloudHoleFilling_ReconstructRGBDxDy "Build the best PointCloudHoleFilling executable?" ON)
if(PointCloudHoleFilling_ReconstructRGBDxDy)
    ADD_EXECUTABLE(ReconstructRGBDxDy ReconstructRGBDxDy.cpp)
    TARGET_LINK_LIBRARIES(ReconstructRGBDxDy ${PointCloudHoleFilling_libraries} gomp)
    INSTALL( TARGETS ReconstructRGBDxDy RUNTIME DESTINATION ${INSTALL_DIR} )
endif()
