////////////////////////////////////////////////////////////////////////
// Terry Griffin
// COMP.5230 Computer Vision I
// Spring 2018
//
// threshold_selection finds the optimal threshold value for separating
// the lung tissue from the surrounding body mass.
//
////////////////////////////////////////////////////////////////////////

#ifndef _THRESHOLD_ROUTINES_
#define _THRESHOLD_ROUTINES_

#include <itkImageRegionConstIterator.h>

// compute the mean voxel values above and below the given threshold
template<typename ImageType>
void threshold_means(typename ImageType::Pointer& image, double T,
		     double *low_mean, double *high_mean)
{
    int low_count = 0;
    int high_count = 0;
    double low_sum = 0;
    double high_sum = 0;


    typedef itk::ImageRegionConstIterator<ImageType> ConstIteratorType;

    ConstIteratorType it(image, image->GetRequestedRegion());

    it.GoToBegin();
    for (  ; !it.IsAtEnd(); ++it)
    {
	if (it.Get() < T)
	{
	    low_count++;
	    low_sum += it.Get();
	}
	else
	{
	    high_count++;
	    high_sum += it.Get();
	}
    }
    *low_mean = (low_count > 0) ? low_sum / low_count : 0.0;
    *high_mean = (high_count > 0) ?  high_sum / high_count : 0.0;
}


// find the optimum threshold value
template<typename ImageType>
double threshold_selection(typename ImageType::Pointer& image)
{
  double T = -500.0;
  double old_T = T + 100;
  double low_mean;
  double high_mean;
  double diff = old_T - T;
  int iteration = 0;
  
  while (fabs(diff) > 5)
  {
      iteration++;
      threshold_means<ImageType>(image, T, &low_mean, &high_mean);
      old_T = T;
      T = (high_mean + low_mean) / 2.0;
      diff = old_T - T;
      std::cout << "iteration: " << iteration <<
	  " T: " << T <<
	  " high_mean: " << high_mean <<
	  " low_mean: " << low_mean << std::endl;
  }
  return T;
}

#endif
