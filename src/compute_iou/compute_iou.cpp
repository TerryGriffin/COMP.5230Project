////////////////////////////////////////////////////////////////////////
// Terry Griffin
// COMP.5230 Computer Vision I
// Spring 2018
//
// compute_iou computes the intersection over union, specificity, and
// sensitivity given a groudn truth image and a mask image. Images
// containing the false positive and false negative voxels are optionally
// generated.
//
////////////////////////////////////////////////////////////////////////

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

#include <string>
#include <map>
#include <vector>

#include <itkutils.h>

int main( int argc, char* argv[] )
{
    bool intermediate_images = false;
    
    if( argc < 3 )
    {
	std::cerr << "Usage: "<< std::endl;
	std::cerr << argv[0];
	std::cerr << "<GroundTruthFilename> <TestFilename> [intermediate_images] " << std::endl;
	return EXIT_FAILURE;
    }
    int index = 1;
    std::string inputFilenameTruth = argv[index++];
    std::string inputFilenameTest = argv[index++];
    std::string prefix_filename;

    for (;index < argc;++index)
    {
	if (strcmp(argv[index],"intermediate_images") == 0)
	    intermediate_images = true;
    }

    std::size_t dotPos = inputFilenameTest.find_last_of(".");
    if (dotPos != std::string::npos)
    {
	prefix_filename = inputFilenameTest.substr(0,dotPos);
    }
    else
    {
	prefix_filename = inputFilenameTest;
    }
    
    const unsigned int Dimension = 3;
    typedef unsigned short LabelType;
    typedef itk::Image<LabelType, Dimension> ImageType;
    ImageType::Pointer imageTruth;
    ImageType::Pointer imageTest;
    
    try
    {
	imageTruth = ReadMHD<ImageType>(inputFilenameTruth);
	imageTest = ReadMHD<ImageType>(inputFilenameTest);
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error reading input image: " << error << std::endl;
	return false;
    }

    typename ImageType::RegionType region;
    typename ImageType::SpacingType spacing;
    typename ImageType::PointType origin;
    ImageType::Pointer imageFalsePos;
    ImageType::Pointer imageFalseNeg;
    typedef itk::ImageRegionIteratorWithIndex<ImageType> IterType;
    int intersection_count = 0;
    int union_count = 0;
    int fn_count = 0;
    int fp_count = 0;
    int tn_count = 0;
    int tp_count = 0;
    int total_pixels = 0;
    double iou;
    
    region=imageTruth->GetLargestPossibleRegion();
    spacing=imageTruth->GetSpacing();
    origin=imageTruth->GetOrigin();

    if (intermediate_images)
    {
	imageFalsePos = ImageType::New();
	imageFalseNeg = ImageType::New();
	imageFalsePos->SetRegions(region);
	imageFalsePos->SetOrigin(origin);
	imageFalsePos->SetSpacing(spacing);
	imageFalsePos->Allocate();
	imageFalsePos->FillBuffer(0);
	imageFalseNeg->SetRegions(region);
	imageFalseNeg->SetOrigin(origin);
	imageFalseNeg->SetSpacing(spacing);
	imageFalseNeg->Allocate();
	imageFalseNeg->FillBuffer(0);
    }

    IterType iter(imageTruth,region);
    bool labelA;
    bool labelB;
    ImageType::IndexType iterIndex;
    for (iter.GoToBegin();!iter.IsAtEnd();++iter)
    {
	total_pixels++;
	iterIndex = iter.GetIndex();

	labelA = iter.Get() != 0;
	labelB = imageTest->GetPixel(iterIndex) != 0;

	if (labelA)
	    tp_count++;
	else
	    tn_count++;
	
	if (labelA && labelB)
	    intersection_count++;
	if (labelA || labelB)
	    union_count++;
	if (labelA && !labelB)
	    fn_count++;
	if (!labelA && labelB)
	    fp_count++;

	if (intermediate_images)
	{
	    if (labelA && !labelB)
		imageFalsePos->SetPixel(iterIndex,1);
	    if (!labelA && labelB)
		imageFalseNeg->SetPixel(iterIndex,2);
	}
    }

    iou = (double) intersection_count / (double) union_count;
    std::cout << basename(inputFilenameTest.c_str())
	      << ", iou: " << iou
	      << ", intersection_count: " << intersection_count
	      << ", union_count: " << union_count
	      << ", true_pos: " << tp_count
	      << ", true_neg: " << tn_count 
	      << ", false_pos: " << fn_count
	      << ", false_neg: " << fp_count
	      << ", specificity: " << (double) tn_count / (tn_count + fp_count)
	      << ", sensitivity: "<< (double) tp_count / (tp_count + fn_count)
	      << std::endl;

    if (intermediate_images)
    {
	std::string filename;

	filename = prefix_filename + "_false_pos.mhd";
	WriteMHD<ImageType>(
	    imageFalsePos,
	    filename);
	filename = prefix_filename + "_false_neg.mhd";
	WriteMHD<ImageType>(
	    imageFalseNeg,
	    filename);
    }
    return EXIT_SUCCESS;
}

