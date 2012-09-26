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
#include <PatchBasedInpainting/Drivers/LidarInpaintingTextureVerification.hpp>

// SmallHoleFiller
#include <SmallHoleFiller/SmallHoleFiller.h>

// ITK
#include "itkImage.h"
#include "itkImageFileReader.h"

/** This program takes a ptx, mask, and completed RGBDxDy image and reconstructs the final point cloud.
  * This functionality if included in PointCloudHoleFilling, but that full procedure (including the inpainting)
  * can take a long time to run.
  */
int main(int argc, char *argv[])
{
  // Verify arguments
  if(argc != 5)
  {
    std::cerr << "Required arguments: PointCloud.ptx imageMask.mask RGBDxDy.mha outputPrefix" << std::endl;
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
  std::string RGBDxDyFileName = argv[3];
  std::string outputPrefix = argv[4];

  // Output arguments
  std::cout << "Reading ptx: " << ptxFileName << std::endl;
  std::cout << "Reading mask: " << maskFileName << std::endl;
  std::cout << "RGBDxDyFileName: " << RGBDxDyFileName << std::endl;
  std::cout << "Output prefix: " << outputPrefix << std::endl;

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

  ptxImage.WritePointCloud(std::string("Original.vtp"));

  ///////////// Fill invalid pixels in the PTX grid /////////////

  // Find the invalid pixels
  PTXImage::MaskImageType::Pointer invalidMaskImage = PTXImage::MaskImageType::New();
  ptxImage.CreateValidityImage(invalidMaskImage);
  Mask::Pointer invalidMask = Mask::New();
  PTXImage::MaskImageType::PixelType holeValue = 0;
  invalidMask->CreateFromImage(invalidMaskImage.GetPointer(), holeValue);

  PTXImage::RGBDImageType::Pointer rgbdImage = PTXImage::RGBDImageType::New();
  ptxImage.CreateRGBDImage(rgbdImage.GetPointer());
  ITKHelpers::WriteImage(rgbdImage.GetPointer(), "RGBD.mha");

  SmallHoleFiller<PTXImage::RGBDImageType> smallHoleFiller(rgbdImage.GetPointer(), invalidMask);
  //smallHoleFiller.SetWriteIntermediateOutput(true);
  unsigned int kernelRadius = 1;
  unsigned int downsampleFactor = 1;
  smallHoleFiller.SetKernelRadius(kernelRadius);
  smallHoleFiller.SetDownsampleFactor(downsampleFactor);
  smallHoleFiller.Fill();

  ITKHelpers::WriteImage(smallHoleFiller.GetOutput(), "Valid.mha");

  ptxImage.SetAllPointsToValid(); // This call must come before ReplaceRGBD, because the values are only replaced for valid pixels!

  ptxImage.ReplaceRGBD(smallHoleFiller.GetOutput());

//  ptxImage.WritePTX(std::string("Valid.ptx"));

  ptxImage.WritePointCloud(std::string("Valid.vtp"));

  // Read the RGBDxDy image
  typedef itk::Image<itk::CovariantVector<float, 5>, 2> RGBDxDyImageType;
  typedef itk::ImageFileReader<RGBDxDyImageType> RGBDxDyReaderType;
  RGBDxDyReaderType::Pointer rgbDxDyReader = RGBDxDyReaderType::New();
  rgbDxDyReader->SetFileName(RGBDxDyFileName);
  rgbDxDyReader->Update();

  RGBDxDyImageType* rgbDxDyImage = rgbDxDyReader->GetOutput();

  ///////////// Assemble the result /////////////
  // Extract inpainted depth gradients
  std::vector<unsigned int> depthGradientChannels = {3, 4};
  typedef itk::Image<itk::CovariantVector<float, 2>, 2> GradientImageType;
  GradientImageType::Pointer inpaintedDepthGradients = GradientImageType::New();
  ITKHelpers::ExtractChannels(rgbDxDyImage, depthGradientChannels,
                              inpaintedDepthGradients.GetPointer());
  ITKHelpers::WriteImage(inpaintedDepthGradients.GetPointer(), "InpaintedDepthGradients.mha");

  // Extract inpainted RGB image
  std::vector<unsigned int> rgbChannels = {0, 1, 2};
  RGBImageType::Pointer inpaintedRGBImage = RGBImageType::New();
  ITKHelpers::ExtractChannels(rgbDxDyImage, rgbChannels,
                              inpaintedRGBImage.GetPointer());
  ITKHelpers::WriteImage(inpaintedRGBImage.GetPointer(), "InpaintedRGB.png");

  // Poisson filling
  typedef PTXImage::DepthImageType DepthImageType;

  DepthImageType::Pointer depthImage = DepthImageType::New();
  ptxImage.CreateDepthImage(depthImage);

  DepthImageType::Pointer inpaintedDepthImage = DepthImageType::New();
  PoissonEditing<float>::FillScalarImage(depthImage.GetPointer(), mask,
                                         inpaintedDepthGradients.GetPointer(),
                                         inpaintedDepthImage.GetPointer());
  ITKHelpers::WriteImage(inpaintedDepthImage.GetPointer(), "ReconstructedDepth.mha");

  // Assemble and write output
  PTXImage filledPTX = ptxImage;
  filledPTX.SetAllPointsToValid();
  filledPTX.ReplaceDepth(inpaintedDepthImage);
  filledPTX.ReplaceRGB(inpaintedRGBImage);

  std::stringstream ssPTXOutput;
  ssPTXOutput << outputPrefix << ".ptx";
  filledPTX.WritePTX(ssPTXOutput.str());

  std::stringstream ssVTPOutput;
  ssVTPOutput << outputPrefix << ".vtp";
  filledPTX.WritePointCloud(ssVTPOutput.str());

  return EXIT_SUCCESS;
}
