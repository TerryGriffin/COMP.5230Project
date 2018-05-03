////////////////////////////////////////////////////////////////////////
// Terry Griffin
// COMP.5230 Computer Vision I
// Spring 2018
//
// extractslice creates an image file for one slice of a CT volume.
//
// Run as: extractslice <input filename> <dimension> [slice#] <outputfilename>
//
// The dimension is:
//    0 - Slice along X axis
//    1 - Slice along Y axis
//    2 - Slice along Z axis
//
// If the slice# is ommited the center slice is used by default.
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
    if( argc < 4 )
    {
	std::cerr << "Usage: "<< std::endl;
	std::cerr << argv[0];
	std::cerr << " <InputFileName> <D> [slice#] <OutputFileName>";
	std::cerr << std::endl;
	return EXIT_FAILURE;
    }
    int index = 1;
    const char *inputFilename = argv[index++];
    int direction = ::atoi(argv[index++]);
    int slice = -1;
    if (argc > 4)
    {
	slice = ::atoi(argv[index++]);
    }
    const char *outputFilename = argv[index++];
    
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
    
    const ImageType::SpacingType& imageSpacing = image->GetSpacing();
    for (int i=0;i<3;++i)
	spacing[i] = imageSpacing[i];
    ImageType::RegionType region = image->GetLargestPossibleRegion();
    ImageType::SizeType imageSize = region.GetSize();

    if (slice == -1)
    {
	slice = imageSize[direction] / 2;
    }

    typedef itk::Image<PixelType, 2> SliceType;
    typedef itk::Image<unsigned short,2> OutputImageType;
    typedef itk::ImageFileWriter<OutputImageType> WriterType;
    WriterType::Pointer writer = WriterType::New();
    itk::ImageIORegion outputRegion;
    typedef itk::ExtractImageFilter<ImageType,SliceType> ExtractFilterType;
    ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
    typedef itk::RescaleIntensityImageFilter<SliceType,OutputImageType> RescalerType;

    RescalerType::Pointer intensityRescaler = RescalerType::New();
    intensityRescaler->SetInput(extractFilter->GetOutput());
    //intensityRescaler->SetOutputMinimum(0);
    //intensityRescaler->SetOutputMaximum(255);
    
    extractFilter->SetInput(image);
    ImageType::RegionType desiredRegion;
    ImageType::IndexType start = region.GetIndex();
    imageSize[direction]=0;
    start[direction]=slice;
    desiredRegion.SetSize(imageSize);
    desiredRegion.SetIndex(start);
    extractFilter->SetExtractionRegion(desiredRegion);
    extractFilter->SetDirectionCollapseToSubmatrix();

    writer->SetFileName(outputFilename);
    writer->SetInput(intensityRescaler->GetOutput());
    //writer->SetIORegion(outputRegion);
    try
    {
	writer->Update();
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error: " << error << std::endl;
	return false;
    }
    

    return EXIT_SUCCESS;
}

