cmake_minimum_required(VERSION 2.6)

PROJECT(PointCloudHoleFilling)
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

SET(ROOT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Where to copy executables when 'make install' is run
SET(INSTALL_DIR ${CMAKE_INSTALL_PREFIX} )
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE) #fix the dynamic linking error (RPATH) that would occur without this

# This is needed for shared_ptr and the trick using enable_if and if_fundamental to allow scalars to be treated as the 0th component of a vector.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")

# Setup include directories for Qt
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# VTK
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

if( "${VTK_MAJOR_VERSION}" LESS 6 )
  MESSAGE(FATAL_ERROR "You must build this code with VTK >= 6.0!")
endif( "${VTK_MAJOR_VERSION}" LESS 6 )

# ITK
FIND_PACKAGE(ITK REQUIRED ITKCommon ITKIOImageBase ITKIOPNG ITKIOMeta
ITKImageIntensity ITKImageFeature ITKMathematicalMorphology ITKBinaryMathematicalMorphology ITKDistanceMap)
INCLUDE(${ITK_USE_FILE})

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

# Submodules
if(NOT TARGET PatchBasedInpainting)
  add_subdirectory(PatchBasedInpainting)
  include_directories(${PatchBasedInpainting_includes})
endif()

if(NOT TARGET PoissonEditing)
  add_subdirectory(PoissonEditing)
include_directories(${PoissonEditing_includes})
endif()

if(NOT TARGET PTXTools)
  add_subdirectory(PTXTools)
  include_directories(${PTXTools_includes})
endif()


if( (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PatchBasedInpainting/.git") OR
(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PoissonEditing/.git") OR
(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/PTXTools/.git") )
  message( SEND_ERROR "At least one of the git submodules is not available.
                       Please run git submodule update --init --recursive")
endif()

ADD_EXECUTABLE(PointCloudHoleFilling PointCloudHoleFilling.cpp)
TARGET_LINK_LIBRARIES(PointCloudHoleFilling PatchBasedInpainting ${VTK_LIBRARIES} ${ITK_LIBRARIES}
      ${QT_LIBRARIES} Mask gomp)
INSTALL( TARGETS PointCloudHoleFilling RUNTIME DESTINATION ${INSTALL_DIR} )
