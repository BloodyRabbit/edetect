/** @file
 * @brief Declaration of CudaConvolution2dFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__
#define CUDA__CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__

#include "IConvolution2dFilter.hxx"

/**
 * @brief Applies full 2D discrete convolution
 *   to the image.
 *
 * @author Jan Bobek
 */
class CudaConvolution2dFilter
: public IConvolution2dFilter
{
public:
    /// The largest supported radius.
    static const unsigned int MAX_RADIUS = 32;

    /// @copydoc IConvolution2dFilter::setKernel(const float*, unsigned int)
    void setKernel( const float* kernel, unsigned int radius );

protected:
    /// @copydoc IConvolution2dFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_CONVOLUTION_2D_FILTER_HXX__INCL__ */
