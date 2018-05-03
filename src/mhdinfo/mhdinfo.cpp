////////////////////////////////////////////////////////////////////////
// Terry Griffin
// COMP.5230 Computer Vision I
// Spring 2018
//
// mhdinfo prints out the size and shape parameters for a MetaImage file
//
////////////////////////////////////////////////////////////////////////
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>
#include <itkRescaleIntensityImageFilter.h>
#include <string>

#include <itkutils.h>

int main( int argc, char* argv[] )
{
    if( argc < 2 )
    {
	std::cerr << "Usage: "<< std::endl;
	std::cerr << argv[0];
	std::cerr << " <InputFileName>";
	std::cerr << std::endl;
	return EXIT_FAILURE;
    }
    int index = 1;
    const char *inputFilename = argv[index++];
    
    double spacing[3];
    const unsigned int Dimension = 3;
    typedef unsigned short PixelType;
    typedef itk::Image<PixelType, Dimension> ImageType;
    ImageType::Pointer image;
    
    try
    {
	image = ReadMHD<ImageType>(inputFilename);
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error: " << error << std::endl;
	return false;
    }
    ImageType::RegionType region = image->GetLargestPossibleRegion();
    ImageType::SizeType imageSize = region.GetSize();
    ImageType::DirectionType direction = image->GetDirection();
    
    std::cout << inputFilename
	      << ", " << image->GetOrigin()
	      << ", " << image->GetSpacing();
    std::cout << ",[";
    bool first = true;
    for (int row=0;row<3;++row)
      for (int col=0;col<3;++col)
      {
	  if (!first)
	  {
	      std::cout << ", ";
	  }
	  first = false;
	  std::cout << direction[row][col];
      }
    std::cout << "]";
    std::cout
	 << ", " << imageSize
	 << std::endl;
    return EXIT_SUCCESS;
}

