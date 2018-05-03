////////////////////////////////////////////////////////////////////////
// Terry Griffin
// COMP.5230 Computer Vision I
// Spring 2018
//
// The routines in this file are used to separate the two lungs in
// the lung mask
//
////////////////////////////////////////////////////////////////////////


#ifndef _SPLIT_LUNGS_H_
#define _SPLIT_LUNGS_H_

#include <iostream>
#include <limits>
#include <unistd.h>

#include <itkutils.h>
#include <itkRegionOfInterestImageFilter.h>

// this is the cost funtion for the dynamic programming max cost
// path algorithm used to find a path between the two lungs
template < class ImageType, class MaskType>
    double cost(double *memo, short *splitMap,  int x, int y,int z,
		int minX, int maxX, int minY, int maxY,
		int sizeX, int sizeY,
	      ImageType  image,
	      MaskType  mask)
		    
{
    size_t arrayIndex = y*sizeX + x;
    
    //std::cout << "cost(" << x << ", " << y << ", " << z << ") " << std::endl;
    if (y > maxY || y < minY)
    {
	//std::cout << "outside y, returning 0" << std::endl;
	return 0;
    }
    if (x > maxX || x < minX)
    {
	//std::cout << "outside x, returning 0" << std::endl;
	return 0;
    }
    
    if (memo[arrayIndex] != std::numeric_limits<double>::min())
    {
	//std::cout << "found cost, returning " << memo[arrayIndex] << std::endl;
	return memo[arrayIndex];
    }
    double leftCost;
    double rightCost;
    double forwardCost;
    double cellCost;
    double totalCost;

 
    typename MaskType::ObjectType::IndexType maskIndex;
    typename ImageType::ObjectType::IndexType imageIndex;

    maskIndex[0] = x;
    maskIndex[1] = y;
    maskIndex[2] = z;
    imageIndex[0] = x;
    imageIndex[1] = y;
    imageIndex[2] = z;

    if (mask->GetPixel(maskIndex) == 0)
    {
	// the pixel is part of the mask
	cellCost = 10000;
    }
    else
    {
	// offset the grayscal value to make sure all costs are positive
	cellCost = image->GetPixel(imageIndex) + 2000;
    }

    // recurrance step
    leftCost = cost(memo,splitMap,x-1,y+1,z,minX,maxX,minY,maxY,
		    sizeX,sizeY,image,mask);
    rightCost =  cost(memo,splitMap,x+1,y+1,z,minX,maxX,minY,maxY,
		      sizeX,sizeY,image,mask);
    forwardCost =  cost(memo,splitMap,x,y+1,z,minX,maxX,minY,maxY,
			sizeX,sizeY,image,mask);

    // decide the best path forward
    if (forwardCost >= rightCost && forwardCost >= leftCost)
    {
	//std::cout << "moving forward ";
	totalCost = cellCost+forwardCost;
	splitMap[arrayIndex] = 1;
    }
    else if (leftCost >= rightCost)
    {
	//std::cout << "moving left ";
	totalCost = cellCost+leftCost;
	splitMap[arrayIndex] = 2;
    }
    else
    {
	//std::cout << "moving right ";
	totalCost = cellCost+rightCost;
	splitMap[arrayIndex] = 3;
    }

    // update the memoization array
    memo[arrayIndex] = totalCost;
    //std::cout << "returning " << totalCost << std::endl;
    return totalCost;
}

// uses the max cost path algorithm to find a path between the two lungs
template < typename ImageType, typename MaskType>
void split_slice(double *memo, int *split, short *splitMap,
		 int sizeX, int sizeY, int z, 
		 ImageType image,
		 MaskType mask)
		    
{
    
    size_t index;
    double splitCost;
    typename MaskType::ObjectType::IndexType maskIndex;

    // find the max cost path. the splitMap will hold the data needed
    // to reconstruct the path
    splitCost = cost<ImageType,MaskType>(memo,splitMap,sizeX/2,0,z,
					 (int)sizeX *0.25,(int) sizeX * 0.75,
					 0, sizeY-1,
					 sizeX, sizeY, image,mask);

    // fill in the split array from the map generated
    split[0] = sizeX / 2;
    for (int y=1;y<sizeY;++y)
    {
	index = y * sizeX + split[y-1];
	if (splitMap[index] == 1)
	{
	    split[y] = split[y-1];
	}
	else if (splitMap[index] == 2)
	{
	    split[y] = split[y-1]-1;
	}
	else if (splitMap[index] == 3)
	{
	    split[y] = split[y-1]+1;

	}
	else
	{
	    std::cerr << "Unexpected value in splitMap" << std::endl;
	}
    }

    // reinitialized the memoization data for the next slice
    for (int i=0;i<sizeX*sizeY;++i)
    {
	memo[i] = std::numeric_limits<double>::min();
	splitMap[i] = 0;
    }

    // If a mask pixel had to change, adjust the memoization values so
    // that the next slice includes a path adjacent to the changed pixels.
    // This prevents the two sides from being connected in 3D
    maskIndex[2] = z;
    for (int y=0;y<sizeY;++y)
    {
	maskIndex[1] = y;
	maskIndex[0] = split[y];

	//std::cout << "split[" << y << "] = " << split[y] << std::endl;

	// testing
	//mask->SetPixel(maskIndex,100);
	if (mask->GetPixel(maskIndex) != 0)
	{
	    mask->SetPixel(maskIndex,0);
	    maskIndex[0] = maskIndex[0] -1;
	    mask->SetPixel(maskIndex,0);
	    maskIndex[0] = maskIndex[0] + 2;
	    mask->SetPixel(maskIndex,0);

	    index = y*sizeX;
	    for (int x=0;x<split[y]-1;++x)
	    {
		memo[index++] = -1e5;
	    }
	    index = y*sizeX + split[y]+2;
	    for (int x=split[y]+2;x<sizeX;++x)
	    {
		memo[index++] = -1e5;
	    }

	}
    }
}

// split the lungs on a per slice basis. The path found for each slice
// is used as the starting point for the next slice.
template < typename ImageType, typename MaskType>
    void split_lungs(ImageType image,
		     MaskType mask,
		     typename ImageType::ObjectType::SizeType imageSize)
{
    // convenience vars to avoid indexing bugs
    int sizeX = imageSize[0];
    int sizeY = imageSize[1];
    int sizeZ = imageSize[2];
    
    int slicePixels = sizeX * sizeY;
    int *lastSplit = new int[sizeY];
    double *memo = new double[slicePixels];
    short *splitMap = new short[slicePixels];

    std::cout << "starting split_lungs " << std::endl;
    // initialize last split to center of image
    int midX = sizeX / 2;
    for (int y=0;y<sizeY;++y)
    {
	lastSplit[y] = midX;
    }

    for (int i=0;i<sizeX*sizeY;++i)
    {
	memo[i] = std::numeric_limits<double>::min();
	splitMap[i] = 0;
    }

    
    typename ImageType::ObjectType::IndexType pixelIndex;
    bool needNewSplit;
    // process each slice

    for (int z=0;z<sizeZ;++z)
    {
	needNewSplit = false;
	pixelIndex[2] = z;
	// check if last split works for this slice
	for (int y=0;y<sizeY;++y)
	{
	    pixelIndex[1] = y;
	    pixelIndex[0] = lastSplit[y];
	    if (mask->GetPixel(pixelIndex) != 0)
	    {
		needNewSplit = true;
		break;
	    }
	}

	// only call split_slice if the current path crosses a lung
	// mask voxel
	if (needNewSplit)
	{
	    //std::cout << "split needed for splice " << z << std::endl;
	    split_slice(memo,lastSplit,splitMap,sizeX,sizeY,z,
					    image,mask);
	}
    }

    delete [] lastSplit;
    delete [] memo;
    delete [] splitMap;

    std::cout << "finished split_lungs " << std::endl;    
}



template<typename ImageType>
ImageType separate_lungs(ImageType mask, int totalPixels)
{
    typedef typename ImageType::ObjectType::PixelType LabelType;
    typedef itk::ConnectedComponentImageFilter<typename ImageType::ObjectType, typename ImageType::ObjectType> ConnectedComponentFilterType;
    typedef itk::LabelShapeKeepNObjectsImageFilter<typename ImageType::ObjectType> KeepComponentsFilterType;
    typedef itk::ShapeLabelObject<LabelType,3> ShapeLabelObjectType;
    typedef itk::LabelMap<ShapeLabelObjectType> LabelMapType;
    typedef itk::LabelImageToShapeLabelMapFilter<typename ImageType::ObjectType,LabelMapType> LabelToShapeMapFilterType;

    typename ConnectedComponentFilterType::Pointer ccFilter;
    typename KeepComponentsFilterType::Pointer keepComponentsFilter;
    typename LabelToShapeMapFilterType::Pointer labelToShapeMapFilter;
    
    int count = 1;
    bool done = false;
    int lungsFound=0;
    int erosionSteps=0;
	    
    for (count=0;count<10 && lungsFound != 2;++count)
    {
	lungsFound = 0;
	std::cout << "separate_lungs step " << count+1 << std::endl;
	ccFilter = ConnectedComponentFilterType::New();
	ccFilter->SetInput(mask);
	ccFilter->SetFullyConnected(true);
	ccFilter->Update();
	if (ccFilter->GetObjectCount() > 1)
	{
	    keepComponentsFilter = KeepComponentsFilterType::New();
	    keepComponentsFilter->SetInput(ccFilter->GetOutput());
	    keepComponentsFilter->SetBackgroundValue(0);
	    keepComponentsFilter->SetNumberOfObjects(2);
	    keepComponentsFilter->SetAttribute(KeepComponentsFilterType::LabelObjectType::NUMBER_OF_PIXELS);

	    keepComponentsFilter->Update();

	    labelToShapeMapFilter = LabelToShapeMapFilterType::New();
	    labelToShapeMapFilter->SetInput(keepComponentsFilter->GetOutput());
	    labelToShapeMapFilter->Update();

	    LabelMapType *labelMap = labelToShapeMapFilter->GetOutput();
	    std::cout << "number of labels: " << labelMap->GetNumberOfLabelObjects()
		      << std::endl;

	    for (unsigned int i=0;i<labelMap->GetNumberOfLabelObjects();++i)
	    {
		ShapeLabelObjectType *labelObject = labelMap->GetNthLabelObject(i);
		LabelType label = labelObject->GetLabel();
	
		if (labelObject->GetNumberOfPixels() >= 0.01 * totalPixels)
		{
		    std::cout << "  " << label << " BoundingBox: "
			      << labelObject->GetBoundingBox() << std::endl;
		    lungsFound++;
		}
	    }
	    if (lungsFound == 2)
	    {
		mask = keepComponentsFilter->GetOutput();
	    }
	}
	else
	{
	    lungsFound = 1;
	}
	if (lungsFound != 2)
	{
	    erosionSteps++;
	    std::cout << "staring erosion step " << erosionSteps << std::endl;
	    mask = erode_image(mask,1);
	    std::cout << "finshed erosion step " << erosionSteps << std::endl;
	}
    }

    for (int i=0;i<erosionSteps;++i)
    {
	sleep(10);
	std::cout << "starting dilate step " << i+1  << std::endl;
	mask = dilate_image(mask,1);
	std::cout << "finshed dilate step " << i+1  << std::endl;
    }
    return mask;
}

// split_mask creates left and right lung masks from the complete mask. This
// is done based on the bounding boxes of the two components
template <typename ImageType>
void split_mask(ImageType  mask,
		ImageType& leftLungMask,
		ImageType& rightLungMask)
{
    typedef typename ImageType::ObjectType::PixelType LabelType;
    typedef itk::ConnectedComponentImageFilter<typename ImageType::ObjectType, typename ImageType::ObjectType> ConnectedComponentFilterType;
    typedef itk::ShapeLabelObject<LabelType,3> ShapeLabelObjectType;
    typedef itk::LabelMap<ShapeLabelObjectType> LabelMapType;
    typedef itk::LabelImageToShapeLabelMapFilter<typename ImageType::ObjectType,LabelMapType> LabelToShapeMapFilterType;

    typedef itk::RegionOfInterestImageFilter<typename ImageType::ObjectType, typename ImageType::ObjectType> ROIFilterType;
    
    typename ImageType::ObjectType::RegionType leftRegion;
    typename ImageType::ObjectType::RegionType rightRegion;
    LabelType leftLabel;
    LabelType rightLabel;
    typename ConnectedComponentFilterType::Pointer ccFilter;
    typename LabelToShapeMapFilterType::Pointer labelToShapeMapFilter;

    // find the two components
    ccFilter = ConnectedComponentFilterType::New();
    ccFilter->SetInput(mask);
    ccFilter->SetFullyConnected(true);
    ccFilter->Update();

    // make sure we start with two components
    if (ccFilter->GetObjectCount() != 2)
    {
	std::cerr << "In split_mask, expected 2 components, found " <<
	    ccFilter->GetObjectCount() << std::endl;
	return;
    }
    
    labelToShapeMapFilter = LabelToShapeMapFilterType::New();
    labelToShapeMapFilter->SetInput(ccFilter->GetOutput());
    labelToShapeMapFilter->Update();

    LabelMapType *labelMap = labelToShapeMapFilter->GetOutput();
    ShapeLabelObjectType *labelObject;

    // get the bounding box for each component
    labelObject = labelMap->GetNthLabelObject(0);
    leftRegion = labelObject->GetBoundingBox();
    leftLabel = labelObject->GetLabel();
    labelObject = labelMap->GetNthLabelObject(1);
    rightRegion = labelObject->GetBoundingBox();
    rightLabel = labelObject->GetLabel();

    // swap the regions if needed
    if (leftRegion.GetIndex(0) < rightRegion.GetIndex(0))
    {
	// swap
	typename ImageType::ObjectType::RegionType tmpRegion;
	LabelType tmpLabel;
	
	tmpRegion = leftRegion;
	tmpLabel = leftLabel;
	leftRegion = rightRegion;
	leftLabel = rightLabel;
	rightRegion = tmpRegion;
	rightLabel = tmpLabel;
    }

    std::cout << "left region: " << leftRegion << std::endl;
    std::cout << "right region: " << rightRegion << std::endl;
    typename ROIFilterType::Pointer roiFilter;

    // create the two sub images
    roiFilter= ROIFilterType::New();
    roiFilter->SetInput(mask);
    roiFilter->SetRegionOfInterest(leftRegion);
    roiFilter->Update();
    leftLungMask = roiFilter->GetOutput();
    roiFilter= ROIFilterType::New();
    roiFilter->SetInput(mask);
    roiFilter->SetRegionOfInterest(rightRegion);
    roiFilter->Update();
    rightLungMask = roiFilter->GetOutput();
}


#endif // _SPLIT_LUNGS_H_

