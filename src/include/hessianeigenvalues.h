////////////////////////////////////////////////////////////////////////
// Terry Griffin
// COMP.5230 Computer Vision I
// Spring 2018
//
// ComputerEigenValues creates an image where each voxel is a vector
// containing the three Eigen values of the Hessian matrix of the
// given image.
//
//
////////////////////////////////////////////////////////////////////////

#ifndef _HESSIANEIGENVALUES_H_
#define _HESSIANEIGENVALUES_H_

#include <itkImage.h>
#include <itkHessianRecursiveGaussianImageFilter.h>
#include <itkImageDuplicator.h>
#include <itkCastImageFilter.h>
#include <itkSymmetricEigenAnalysis.h>

typedef float ValueType;
const int Dimension = 3;

typedef itk::Vector<ValueType,Dimension> EigenValueVoxType;
typedef itk::Matrix<ValueType,Dimension> HessianMatrixType;
typedef itk::Image<EigenValueVoxType,Dimension> EigenValueImageType;


template <typename ImageType>
EigenValueImageType::Pointer ComputeEigenValues(typename ImageType::Pointer image,
						double sigma)
{
    typedef itk::HessianRecursiveGaussianImageFilter<ImageType> HessianFilterType;
    typedef typename HessianFilterType::OutputPixelType HessianOutputPixelType;
    typedef typename itk::SymmetricEigenAnalysis<HessianMatrixType,EigenValueVoxType> SymmetricEigenAnalysisType;
    typedef typename itk::ImageRegionIteratorWithIndex<typename HessianFilterType::OutputImageType> IterType;
    
    typename HessianFilterType::Pointer hessianFilter = HessianFilterType::New();
    SymmetricEigenAnalysisType eigenAnalysis;
    typename ImageType::RegionType region;
    typename ImageType::SpacingType spacing;
    typename ImageType::PointType origin;
    EigenValueImageType::Pointer eigenValueImage;
    EigenValueVoxType eigenVox;

    // get the area to work with
    region=image->GetLargestPossibleRegion();
    spacing=image->GetSpacing();
    origin=image->GetOrigin();

    // set the parameters of the Hessian filter
    hessianFilter->SetSigma(sigma);
    hessianFilter->SetInput(image);

    // compute the Hessian matrix for the image
    try
    {
	std::cout << "starting HessianFilter" << std::endl;
	hessianFilter->Update();
	std::cout << "finished HessianFilter" << std::endl;
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error running Hessian Filter " << error << std::endl;
    }

    // Allocate the image for the Eigen values
    eigenAnalysis.SetDimension(Dimension);
    
    eigenValueImage = EigenValueImageType::New();
    eigenValueImage->SetRegions(region);
    eigenValueImage->SetOrigin(origin);
    eigenValueImage->SetSpacing(spacing);
    eigenValueImage->Allocate();

    for (int i=0;i<Dimension;++i)
	eigenVox[i] = 0;

    eigenValueImage->FillBuffer(eigenVox);
    IterType iter(hessianFilter->GetOutput(),region);

    int pixelCounter = 0;

    // iterate over the Hessian image and compute the Eigen values
    // at each voxel.
    
    std::cout << "starting eigenvalue computation" << std::endl;
    HessianMatrixType hessian;
    for (iter.GoToBegin();!iter.IsAtEnd();++iter)
    {
	typename HessianFilterType::OutputImageType::IndexType index = iter.GetIndex();
	auto gradients = iter.Get();
	hessian[0][0] = gradients[0];
	hessian[0][1] = gradients[1];
	hessian[1][0] = gradients[1];
	hessian[0][2] = gradients[2];
	hessian[2][0] = gradients[2];
	hessian[1][1] = gradients[3];
	hessian[2][1] = gradients[4];
	hessian[1][2] = gradients[4];
	hessian[2][2] = gradients[5];

	eigenAnalysis.ComputeEigenValues(hessian,eigenVox);
	eigenValueImage->SetPixel(index,eigenVox);

	if (pixelCounter++ % 1000000 == 0)
	{
	    std::cout << "progress" << pixelCounter / 1000000 << std::endl;
	}
    }
    
    std::cout << "finished eigenvalue computation" << std::endl;
    return eigenValueImage;
}

#endif // _HESSIANEIGENVALUES_H_
