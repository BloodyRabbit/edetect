/** @file
 * @brief Declaration of CudaConvolutionFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_CONVOLUTION_FILTER_HXX__INCL__
#define CUDA__CUDA_CONVOLUTION_FILTER_HXX__INCL__

#include "IConvolutionFilter.hxx"

/**
 * @brief Applies full 2D discrete convolution
 *   to the image.
 *
 * @author Jan Bobek
 */
class CudaConvolutionFilter
: public IConvolutionFilter
{
public:
    /// The largest supported radius.
    static const unsigned int MAX_RADIUS = 32;

    /// @copydoc IConvolutionFilter::setKernel(const float*, unsigned int)
    void setKernel( const float* kernel, unsigned int radius );

protected:
    /// @copydoc IConvolutionFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_CONVOLUTION_FILTER_HXX__INCL__ */
