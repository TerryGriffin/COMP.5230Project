////////////////////////////////////////////////////////////////////////
// Terry Griffin
// COMP.5230 Computer Vision I
// Spring 2018
//
// lung_segmentation
// This program performs lung segmentation on a CT scan.
// Usage: 
// lung_segmentation <InputFileName> [options ...]
// Options: 
//    intermediate_images - save images at each step in the process
//    no_gaussian_smoothing - skip the gaussian smoothing step
//    component_stats - print the statistics from the connected component analysys
//
// The input file is expected to be a MetaImage file. The output lung mask
// will have the same file name with "_mask" appended.
//
////////////////////////////////////////////////////////////////////////
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkExtractImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkCurvatureFlowImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkGradientAnisotropicDiffusionImageFilter.h>
#include <itkChangeLabelImageFilter.h>
#include <itkLabelShapeKeepNObjectsImageFilter.h>
#include <itkLabelImageToShapeLabelMapFilter.h>
#include <itkDiscreteGaussianImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkBinaryErodeImageFilter.h>
#include <itkBinaryDilateImageFilter.h>
#include <itkBinaryBallStructuringElement.h>
#include <itkGrayscaleErodeImageFilter.h>
#include <itkGrayscaleDilateImageFilter.h>
#include <itkImageDuplicator.h>

#include <string>
#include <map>
#include <vector>
#include <libgen.h>

#include <itkutils.h>
#include <threshold.h>
#include <split_lungs.h>
#include <hessianeigenvalues.h>

#define SHOW_COMPONENT_STATS 1

const int BACKGROUND_THRESHOLD = -1000;

int main( int argc, char* argv[] )
{
    // read arguments and set operating parameters
    if( argc < 2 )
    {
	std::cerr << "Usage: "<< std::endl;
	std::cerr << argv[0];
	std::cerr << "<InputFileName> [options ...]" << std::endl;
	std::cerr << "\tOptions: " << std::endl;
	std::cerr << "intermediate_images" << std::endl;
	std::cerr << "no_gaussian_smoothing" << std::endl;
	std::cerr << "component_stats" << std::endl;
	std::cerr << std::endl;
	return EXIT_FAILURE;
    }
    int index = 1;
    bool intermediate_images = false;
    bool gaussian_smoothing = true;
    bool component_stats = false;
    
    char *inputFilename = argv[index++];

    for (; index < argc; index++)
    {
	if (strcmp(argv[index],"intermediate_images") == 0)
	{
	    intermediate_images = true;
	}
	else if (strcmp(argv[index],"no_gaussian_smoothing") == 0)
	{
	    gaussian_smoothing = false;
	}
	else if (strcmp(argv[index],"component_stats") == 0)
	{
	    component_stats = true;
	}
	else
	{
	    std::cerr << "unrecognized option: " << argv[index];
	    return -1;
	}
    }

    std::string filename = basename(inputFilename);
    std::string prefix_filename;

    std::cout << "File: " << filename << std::endl;

    std::size_t dotPos = filename.find_last_of(".");
    if (dotPos != std::string::npos)
    {
	prefix_filename = filename.substr(0,dotPos);
    }
    else
    {
	prefix_filename = filename;
    }
	
    if (file_exists(prefix_filename + "_mask.mhd"))
	return 0;

    // define the types required for processing the image
    const unsigned int Dimension = 3;
    typedef float InternalPixelType;
    typedef itk::Image<InternalPixelType, Dimension> InternalImageType;
    InternalImageType::Pointer image;
    InternalImageType::Pointer originalImage;
    typedef unsigned char                            OutputPixelType;
    typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
    typedef itk::CastImageFilter< InternalImageType, OutputImageType >
                                                   CastingFilterType;

    CastingFilterType::Pointer caster = CastingFilterType::New();

    // Read in the image and store in originalImage
    try
    {
	originalImage = ReadMHD<InternalImageType>(inputFilename);
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error: " << error << std::endl;
	return 1;
    }
    std::cout << "image read" << std::endl;

    image = originalImage;

    // extrac the image parameters
    InternalImageType::RegionType region = image->GetLargestPossibleRegion();
    InternalImageType::SizeType imageSize = region.GetSize();
    InternalImageType::DirectionType direction = image->GetDirection();
    unsigned int totalPixels = imageSize[0]*imageSize[1]*imageSize[2];

    // Threshold pixels to remove background.
    typedef itk::ThresholdImageFilter<InternalImageType> BackgroundThresholdFilterType;

    BackgroundThresholdFilterType::Pointer backgroundFilter = BackgroundThresholdFilterType::New();

    backgroundFilter->SetInput(image);
    backgroundFilter->SetOutsideValue(BACKGROUND_THRESHOLD);
    backgroundFilter->ThresholdBelow(BACKGROUND_THRESHOLD);
    backgroundFilter->Update();
    image = backgroundFilter->GetOutput();

    // save the image if requested
    if (intermediate_images)
    {
	try
	{
	    std::string outputFilename;
	    outputFilename = prefix_filename + "_background_removed.mhd";
	    caster->SetInput(image);
	    caster->Update();
	    WriteMHDFromOutput<OutputImageType,CastingFilterType::Pointer>(
		caster,
		outputFilename);
	}
	catch (itk::ExceptionObject& error)
	{
	    std::cerr << "Error writing thresholded image: " << error << std::endl;
	    return 1;
	}
    }
    
    if (gaussian_smoothing)
    {
	// smooth the image if requested
	typedef itk::DiscreteGaussianImageFilter<InternalImageType,InternalImageType> SmoothingFilterType;

	SmoothingFilterType::Pointer smoothingFilter = SmoothingFilterType::New();
	smoothingFilter->SetInput(image);
	smoothingFilter->SetVariance(1.0);

	std::cout << "starting smoothing" << std::endl;
	smoothingFilter->Update();
	std::cout << "finished smoothing" << std::endl;
	image = smoothingFilter->GetOutput();
    }


    // determine the optimal threshold value
    double threshold = threshold_selection<InternalImageType>(image);
    std::cout << "Optimal threshold: " << threshold << std::endl;

    // Threshold the image. This produces a binary image with the lungs
    // at 255 and other voxels at 0
    typedef itk::BinaryThresholdImageFilter<InternalImageType,InternalImageType> ThresholdFilterType;
    ThresholdFilterType::Pointer thresholdFilter = ThresholdFilterType::New();

    thresholdFilter->SetInput(image);
    thresholdFilter->SetOutsideValue(0);
    thresholdFilter->SetInsideValue(255);
    thresholdFilter->SetLowerThreshold(-10000);
    thresholdFilter->SetUpperThreshold(threshold);

    try
    {
	thresholdFilter->Update();
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error thresholding: " << error << std::endl;
	return 1;
    }

    // save the image if requested
    if (intermediate_images)
    {
	try
	{
	    std::string outputFilename;
	    outputFilename = prefix_filename + "_thresh.mhd";
	    caster->SetInput(thresholdFilter->GetOutput());
	    caster->Update();
	    WriteMHDFromOutput<OutputImageType,CastingFilterType::Pointer>(
		caster,
		outputFilename);
	}
	catch (itk::ExceptionObject& error)
	{
	    std::cerr << "Error writing thresholded image: " << error << std::endl;
	    return 1;
	}
    }

    // define types for the connected component analysis
    typedef unsigned short LabelType;
    typedef itk::Image<LabelType, Dimension> ConnectedComponentImageType;
    typedef itk::ConnectedComponentImageFilter<ThresholdFilterType::OutputImageType,ConnectedComponentImageType> ConnectedComponentFilterType;
    typedef itk::ImageDuplicator<ConnectedComponentImageType> DuplicatorType;

    ConnectedComponentImageType::Pointer componentsImage;

    ConnectedComponentFilterType::Pointer ccFilter = ConnectedComponentFilterType::New();

    // perform the connected component analysis
    ccFilter->SetInput(thresholdFilter->GetOutput());
    ccFilter->SetFullyConnected(true);
    ccFilter->Update();

    if (intermediate_images)
    {
	try
	{
	    std::string outputFilename = prefix_filename + "_components.mhd";
	    WriteMHDFromOutput<ConnectedComponentImageType,ConnectedComponentFilterType::Pointer>(
		ccFilter,
		outputFilename);
	}
	catch (itk::ExceptionObject& error)
	{
	    std::cerr << "Error writing thresholded image: " << error << std::endl;
	    return 1;
	}
    }
    
    std::cout << "Number of components: " << ccFilter->GetObjectCount()
	       << std::endl;

    componentsImage = ccFilter->GetOutput();

    // Find the components connected to the corners and centers of the edges.
    // These components make up the area outside the body.
    // Change the labels on these components to 0 to remove them from
    // the mask.
    
    typedef itk::ChangeLabelImageFilter<ConnectedComponentImageType,ConnectedComponentImageType> ChangeLabelFilterType;
    typedef ChangeLabelFilterType::ChangeMapType ChangeLabelMapType;

    ChangeLabelMapType changeLabelMap;
    // get corners and center edge components
    
    unsigned int midpointX = imageSize[0] / 2;
    unsigned int midpointY = imageSize[1] / 2;
    unsigned int midpointZ = imageSize[2] / 2;
    unsigned int maxX = imageSize[0]-1;
    unsigned int maxY = imageSize[1]-1;
    unsigned int maxZ = imageSize[2]-1;

    std::vector<unsigned int> xPoints{0,midpointX,maxX};
    std::vector<unsigned int> yPoints{0,midpointY,maxY};
    std::vector<unsigned int> zPoints{0,midpointZ,maxZ};
    
    
    ConnectedComponentImageType::IndexType pixelIndex;
    unsigned int label;
    
    for (int i=0;i<3;++i)
	for (int j=0;j<3;++j)
	    for (int k=0;k<3;++k)
		// skip center pixel
		if (i!=1 || j!=1 )
		{
		    pixelIndex[0] = xPoints[i];
		    pixelIndex[1] = yPoints[j];
		    pixelIndex[2] = zPoints[k];
		    label = componentsImage->GetPixel(pixelIndex);
		    if (label != 0)
		    {
			changeLabelMap[label] = 0;
			if (component_stats)
			{
			    std::cout << "replacing label " << label << std::endl;
			}
		    }
		}

    ChangeLabelFilterType::Pointer changeLabelFilter = ChangeLabelFilterType::New();
    changeLabelFilter->SetChangeMap(changeLabelMap);
    changeLabelFilter->SetInput(componentsImage);
    changeLabelFilter->SetInPlace(true);
    changeLabelFilter->Update();

    // save the intermediate image if requested
    if (intermediate_images)
    {
	try
	{
	    std::string outputFilename = prefix_filename + "_components2.mhd";
	    WriteMHDFromOutput<ConnectedComponentImageType,ChangeLabelFilterType::Pointer>(
		changeLabelFilter,
		outputFilename);
	}
	catch (itk::ExceptionObject& error)
	{
	    std::cerr << "Error writing thresholded image: " << error << std::endl;
	    return 1;
	}
    }

    componentsImage = changeLabelFilter->GetOutput();

    // This filter will keep the two largest components and add others
    // to the background. This should result in either two lungs, or one
    // large component containing both lungs.
    typedef itk::LabelShapeKeepNObjectsImageFilter<ConnectedComponentImageType> KeepComponentsFilterType;

    KeepComponentsFilterType::Pointer keepComponentsFilter = KeepComponentsFilterType::New();
    keepComponentsFilter->SetInput(componentsImage);
    keepComponentsFilter->SetBackgroundValue(0);
    keepComponentsFilter->SetNumberOfObjects(2);
    keepComponentsFilter->SetAttribute(KeepComponentsFilterType::LabelObjectType::NUMBER_OF_PIXELS);

    keepComponentsFilter->Update();

    // Save the intermediate image if requested.
    if (intermediate_images)
    {
	try
	{
	    std::string outputFilename = prefix_filename + "_components3.mhd";
	    WriteMHDFromOutput<ConnectedComponentImageType,KeepComponentsFilterType::Pointer>(
		keepComponentsFilter,
		outputFilename);
	}
	catch (itk::ExceptionObject& error)
	{
	    std::cerr << "Error writing thresholded image: " << error << std::endl;
	    return 1;
	}
    }

    // Analyse the two largest components. Remove one if it contains less than
    // 1% of the pixels.
    typedef itk::ShapeLabelObject<LabelType,Dimension> ShapeLabelObjectType;
    typedef itk::LabelMap<ShapeLabelObjectType> LabelMapType;
    typedef itk::LabelImageToShapeLabelMapFilter<ConnectedComponentImageType,LabelMapType> LabelToShapeMapFilterType;

    LabelToShapeMapFilterType::Pointer labelToShapeMapFilter = LabelToShapeMapFilterType::New();
    labelToShapeMapFilter->SetInput(keepComponentsFilter->GetOutput());
    labelToShapeMapFilter->Update();

    LabelMapType *labelMap = labelToShapeMapFilter->GetOutput();
    std::cout << "number of labels: " << labelMap->GetNumberOfLabelObjects()
	      << std::endl;
    bool foundTwoLungs = false;

    ChangeLabelMapType removeSmallComponentsMap;
    LabelType singleLabel = 0;
    for (unsigned int i=0;i<labelMap->GetNumberOfLabelObjects();++i)
    {
	ShapeLabelObjectType *labelObject = labelMap->GetNthLabelObject(i);
	LabelType label = labelObject->GetLabel();
	
	if (labelObject->GetNumberOfPixels() < 0.01 * totalPixels)
	{
	    std::cout << "shape " << label << " too small to keep" << std::endl;
	    removeSmallComponentsMap[label] = 0;
	}
	else
	{
	    std::cout << "shape " << label << " large enough to keep" << std::endl;
	    singleLabel = label;
	}

	if (component_stats)
	{
	    std::cout << "Label: "
		      << itk::NumericTraits<LabelMapType::LabelType>::PrintType(labelObject->GetLabel()) << std::endl;
	    std::cout << "    BoundingBox: "
		      << labelObject->GetBoundingBox() << std::endl;
	    std::cout << "    NumberOfPixels: "
		      << labelObject->GetNumberOfPixels() << std::endl;
	    std::cout << "    PhysicalSize: "
		      << labelObject->GetPhysicalSize() << std::endl;
	    std::cout << "    Centroid: "
		      << labelObject->GetCentroid() << std::endl;
	    std::cout << "    NumberOfPixelsOnBorder: "
		      << labelObject->GetNumberOfPixelsOnBorder() << std::endl;
	    std::cout << "    PerimeterOnBorder: "
		      << labelObject->GetPerimeterOnBorder() << std::endl;
	    std::cout << "    FeretDiameter: "
		      << labelObject->GetFeretDiameter() << std::endl;
	    std::cout << "    PrincipalMoments: "
		      << labelObject->GetPrincipalMoments() << std::endl;
	    std::cout << "    PrincipalAxes: "
		      << labelObject->GetPrincipalAxes() << std::endl;
	    std::cout << "    Elongation: "
		      << labelObject->GetElongation() << std::endl;
	    std::cout << "    Perimeter: "
		      << labelObject->GetPerimeter() << std::endl;
	    std::cout << "    Roundness: "
		      << labelObject->GetRoundness() << std::endl;
	    std::cout << "    EquivalentSphericalRadius: "
		      << labelObject->GetEquivalentSphericalRadius() << std::endl;
	    std::cout << "    EquivalentSphericalPerimeter: "
		      << labelObject->GetEquivalentSphericalPerimeter() << std::endl;
	    std::cout << "    EquivalentEllipsoidDiameter: "
		      << labelObject->GetEquivalentEllipsoidDiameter() << std::endl;
	    std::cout << "    Flatness: "
		      << labelObject->GetFlatness() << std::endl;
	    std::cout << "    PerimeterOnBorderRatio: "
		      << labelObject->GetPerimeterOnBorderRatio() << std::endl;
	}
    }

    if (removeSmallComponentsMap.size() == 2)
    {
	std::cerr << "no large components found" << std::endl;
	return -1;
    }
    if (removeSmallComponentsMap.size() == 1)
    {
	ChangeLabelFilterType::Pointer removeSmallComponentsFilter = ChangeLabelFilterType::New();
	removeSmallComponentsFilter->SetChangeMap(removeSmallComponentsMap);
	removeSmallComponentsFilter->SetInput(keepComponentsFilter->GetOutput());
	removeSmallComponentsFilter->SetInPlace(true);
	removeSmallComponentsFilter->Update();
	componentsImage = removeSmallComponentsFilter->GetOutput();
	std::cout << "Single component found" << std::endl;
	foundTwoLungs = false;
    }
    else
    {
	componentsImage = keepComponentsFilter->GetOutput();
	std::cout << "Two components found" << std::endl;
	foundTwoLungs = true;
    }


    // If both lungs are not found as separate components then separate them
    if (!foundTwoLungs)
    {
	// first split the lungs slice by slice
	split_lungs(originalImage,componentsImage,imageSize);
	// next use erosion to separate the 3D volumes if necessary
	componentsImage = separate_lungs(componentsImage,totalPixels);

	if (intermediate_images)
	{
	    try
	    {
		std::string outputFilename = prefix_filename + "_componentsSplit.mhd";
		WriteMHD<ConnectedComponentImageType>(
		    componentsImage,
		    outputFilename);
	    }
	    catch (itk::ExceptionObject& error)
	    {
		std::cerr << "Error writing thresholded image: " << error << std::endl;
		return 1;
	    }
	}
    }

    if (intermediate_images)
    {
	try
	{
	    std::string outputFilename = prefix_filename + "pre_dilate.mhd";
	    WriteMHD<ConnectedComponentImageType>(
		componentsImage,
		outputFilename);
	}
	catch (itk::ExceptionObject& error)
	{
	    std::cerr << "Error writing mask image: " << error << std::endl;
	    return 1;
	}
    }

    // use erosion followed by dilation for opening the lung mask. This
    // will smooth the boundary and fill in internal holes.
    typedef itk::BinaryBallStructuringElement<LabelType,Dimension> StructuringElementType;
    typedef itk::GrayscaleErodeImageFilter<ConnectedComponentImageType,ConnectedComponentImageType,StructuringElementType> GrayscaleErodeFilterType;
    typedef itk::GrayscaleDilateImageFilter<ConnectedComponentImageType,ConnectedComponentImageType,StructuringElementType> GrayscaleDilateFilterType;

    GrayscaleErodeFilterType::Pointer erodeFilter = GrayscaleErodeFilterType::New();
    GrayscaleDilateFilterType::Pointer dilateFilter = GrayscaleDilateFilterType::New();
    StructuringElementType structuringElement;

    structuringElement.SetRadius(3);
    structuringElement.CreateStructuringElement();

    erodeFilter->SetKernel(structuringElement);
    dilateFilter->SetKernel(structuringElement);
    
    dilateFilter->SetInput(componentsImage);
    erodeFilter->SetInput(dilateFilter->GetOutput());

    try
    {
	std::cout << "Starting dilate/erode " << std::endl;
	erodeFilter->Update();
	std::cout << "Finished dilate/erode " << std::endl;
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error from dilate/erode: " << error << std::endl;
	    return 1;
    }
    componentsImage = erodeFilter->GetOutput();
    if (intermediate_images)
    {
	try
	{
	    std::string outputFilename = prefix_filename + "_dilated.mhd";
	    WriteMHD<ConnectedComponentImageType>(
		componentsImage,
		outputFilename);
	}
	catch (itk::ExceptionObject& error)
	{
	    std::cerr << "Error writing thresholded image: " << error << std::endl;
	    return 1;
	}
    }

    // write out the final lung mask
    try
    {
	std::string outputFilename = prefix_filename + "_mask.mhd";
	WriteMHD<ConnectedComponentImageType>(
	    componentsImage,
	    outputFilename);
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error writing mask image: " << error << std::endl;
	return 1;
    }
    
    return 0;
}


