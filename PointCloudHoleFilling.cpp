/*=========================================================================
 *
 *  Copyright David Doria 2012 daviddoria@gmail.com
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// STL
#include <sstream>

// Submodules
#include <Helpers/Helpers.h>
#include <Mask/Mask.h>
#include <ITKHelpers/ITKHelpers.h>

// PoissonEditing
#include <PoissonEditing/PoissonEditing.h>

// PTXTools
#include <PTXTools/PTXReader.h>

// PatchBasedInpainting
#include <PatchBasedInpainting/ImageProcessing/Derivatives.h>
#include <PatchBasedInpainting/Drivers/InpaintingTexture.hpp>

// ITK
#include "itkImage.h"
#include "itkImageFileReader.h"

int main(int argc, char *argv[])
{
  // Verify arguments
  if(argc != 5)
  {
    std::cerr << "Required arguments: PointCloud.ptx imageMask.mask patchHalfWidth output.png" << std::endl;
    std::cerr << "Input arguments: ";
    for(int i = 1; i < argc; ++i)
    {
      std::cerr << argv[i] << " ";
    }
    return EXIT_FAILURE;
  }

  // Parse arguments
  std::string ptxFileName = argv[1];
  std::string maskFileName = argv[2];

  std::stringstream ssPatchHalfWidth;
  ssPatchHalfWidth << argv[3];
  unsigned int patchHalfWidth = 0;
  ssPatchHalfWidth >> patchHalfWidth;

  std::string outputFileName = argv[4];

  // Output arguments
  std::cout << "Reading ptx: " << ptxFileName << std::endl;
  std::cout << "Reading mask: " << maskFileName << std::endl;
  std::cout << "Patch half width: " << patchHalfWidth << std::endl;
  std::cout << "Output: " << outputFileName << std::endl;

  // Read the files
  PTXImage ptxImage = PTXReader::Read(ptxFileName);

  Mask::Pointer mask = Mask::New();
  mask->Read(maskFileName);

  if(mask->GetLargestPossibleRegion() != ptxImage.GetFullRegion())
  {
    std::stringstream ss;
    ss << "PTX and mask must be the same size! PTX is " << ptxImage.GetFullRegion()
       << " and mask is " << mask->GetLargestPossibleRegion();
    throw std::runtime_error(ss.str());
  }

  typedef PTXImage::DepthImageType DepthImageType;
  DepthImageType::Pointer depthImage = DepthImageType::New();
  ptxImage.CreateDepthImage(depthImage);

  typedef itk::Image<itk::CovariantVector<float, 2>, 2> GradientImageType;
  GradientImageType::Pointer depthGradientImage = GradientImageType::New();

  // Not sure if this will work correctly since the Poisson equation needs to use the same operator as was used in the derivative computations.
  // Potentially use the techniqie in ITK_OneShot:ForwardDifferenceDerivatives instead?
  Derivatives::MaskedGradient(depthImage.GetPointer(), mask, depthGradientImage.GetPointer());

  typedef PTXImage::RGBImageType RGBImageType;
  RGBImageType::Pointer rgbImage = RGBImageType::New();
  ptxImage.CreateRGBImage(rgbImage);

  // Construct RGBDxDy image to inpaint
  typedef itk::Image<itk::CovariantVector<float, 5>, 2> RGBDxDyImageType;
  RGBDxDyImageType::Pointer rgbDxDyImage = RGBDxDyImageType::New();
  ITKHelpers::StackImages(rgbImage.GetPointer(), depthGradientImage.GetPointer(), rgbDxDyImage.GetPointer());

  // Inpaint
  const unsigned int numberOfKNN = 100;
  InpaintingTexture(rgbDxDyImage.GetPointer(), mask, patchHalfWidth, numberOfKNN);

  // Extract inpainted depth gradients
  std::vector<unsigned int> depthGradientChannels = {3, 4};
  GradientImageType::Pointer inpaintedDepthGradients = GradientImageType::New();
  ITKHelpers::ExtractChannels(rgbDxDyImage.GetPointer(), depthGradientChannels,
                              inpaintedDepthGradients.GetPointer());

  // Extract inpainted RGB image
  std::vector<unsigned int> rgbChannels = {0, 1, 2};
  RGBImageType::Pointer inpaintedRGBImage = RGBImageType::New();
  ITKHelpers::ExtractChannels(rgbDxDyImage.GetPointer(), rgbChannels,
                              inpaintedRGBImage.GetPointer());

  // Poisson filling
  DepthImageType::Pointer inpaintedDepthImage = DepthImageType::New();
  PoissonEditing<float>::FillScalarImage(depthImage.GetPointer(), mask,
                                         inpaintedDepthGradients.GetPointer(),
                                         inpaintedDepthImage.GetPointer());

  // Assemble and write output
  PTXImage filledPTX = ptxImage;
  filledPTX.SetAllPointsToValid();
  filledPTX.ReplaceDepth(inpaintedDepthImage);
  filledPTX.ReplaceRGB(inpaintedRGBImage);
  filledPTX.WritePointCloud(outputFileName);

  return EXIT_SUCCESS;
}
