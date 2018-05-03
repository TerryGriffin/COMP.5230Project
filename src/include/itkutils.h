////////////////////////////////////////////////////////////////////////
// Terry Griffin
// COMP.5230 Computer Vision I
// Spring 2018
//
// This file contains utility routines for working with ITK images
//
//
////////////////////////////////////////////////////////////////////////
#ifndef _ITKUTILS_H_
#define _ITKUTILS_H_

#include <string>
#include <sys/stat.h>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkGrayscaleErodeImageFilter.h>
#include <itkGrayscaleDilateImageFilter.h>
#include <itkBinaryBallStructuringElement.h>

// determine if a file exists
bool file_exists(const std::string& name)
{
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

// read a MetaImage file and return the image 
template<typename ImageType>
typename ImageType::Pointer ReadMHD(const std::string& filename)
{
    typedef itk::ImageFileReader<ImageType> ReaderType;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(filename);
    reader->Update();
    return reader->GetOutput();
}

// Write a MetaImage file from the output of some filter
template<typename ImageType, typename SourceType>
void WriteMHDFromOutput(SourceType source, const std::string& filename)
{
    typedef itk::ImageFileWriter<ImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(source->GetOutput());
    writer->Update();
}

// Write a MetaImage file given an image
template<typename ImageType>
void WriteMHD(typename ImageType::Pointer image, const std::string& filename)
{
    typedef itk::ImageFileWriter<ImageType> WriterType;
    typename WriterType::Pointer writer = WriterType::New();
    writer->SetFileName(filename);
    writer->SetInput(image);
    writer->Update();
}

// Run a 3D erosion using a ball structuring element with the given radius
template<typename ImageType>
ImageType erode_image(ImageType image, int radius)
{
    typedef itk::BinaryBallStructuringElement<typename ImageType::ObjectType::PixelType,3> StructuringElementType;
    typedef itk::GrayscaleErodeImageFilter<typename ImageType::ObjectType,
	typename ImageType::ObjectType,
	StructuringElementType> ErodeFilterType;

    typename ErodeFilterType::Pointer erode = ErodeFilterType::New();
    StructuringElementType structuringElement;
    structuringElement.SetRadius(radius);
    structuringElement.CreateStructuringElement();

    erode->SetKernel(structuringElement);
    erode->SetInput(image);
    try
    {
	erode->Update();
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error in erode_image: " << error << std::endl;
	return image;
    }
    return erode->GetOutput();
}

// Run a 3D dilation using a ball structuring element with the given radius
template<typename ImageType>
ImageType dilate_image(ImageType image, int radius)
{
    typedef itk::BinaryBallStructuringElement<typename ImageType::ObjectType::PixelType,3> StructuringElementType;
    typedef itk::GrayscaleDilateImageFilter<typename ImageType::ObjectType,
	typename ImageType::ObjectType,
	StructuringElementType> DilateFilterType;

    typename DilateFilterType::Pointer dilate = DilateFilterType::New();
    StructuringElementType structuringElement;
    structuringElement.SetRadius(radius);
    structuringElement.CreateStructuringElement();

    dilate->SetKernel(structuringElement);
    dilate->SetInput(image);
    try
    {
	dilate->Update();
    }
    catch (itk::ExceptionObject& error)
    {
	std::cerr << "Error in dilate_image: " << error << std::endl;
	return image;
    }
    return dilate->GetOutput();
}

#endif // _ITKUTILS_H_
