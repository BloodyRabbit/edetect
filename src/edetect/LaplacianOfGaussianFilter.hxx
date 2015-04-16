/** @file
 * @brief Declaration of class LaplacianOfGaussianFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#ifndef LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__
#define LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__

#include "IGeneratedKernelFilter.hxx"

/**
 * @brief Convolves the image with
 *  the Laplacian-of-Gaussian (LoG) kernel.
 *
 * @author Jan Bobek
 */
template< typename CF >
class LaplacianOfGaussianFilter
: public IGeneratedKernelFilter< CF >
{
public:
    /**
     * @brief Sets radius of the LoG kernel.
     *
     * @param[in] radius
     *   The desired kernel radius.
     */
    void setRadius( unsigned int radius );

protected:
    // Improves readability
    using IGeneratedKernelFilter< CF >::mKernel;
    using IGeneratedKernelFilter< CF >::mFilter;
};

#include "LaplacianOfGaussianFilter.txx"

#endif /* !LAPLACIAN_OF_GAUSSIAN_FILTER_HXX__INCL__ */
