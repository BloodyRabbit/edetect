/** @file
 * @brief Declaration of CudaConvolutionFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_CONVOLUTION_FILTER_HXX__INCL__
#define CUDA__CUDA_CONVOLUTION_FILTER_HXX__INCL__

#include "filters/IConvolutionFilter.hxx"

/**
 * @brief Interface of a CUDA-backed convolution filter.
 *
 * @author Jan Bobek
 */
class ICudaConvolutionFilter
: public IConvolutionFilter
{
public:
    /// The largest supported radius.
    static const unsigned int MAX_RADIUS = 32;
    /// The largest supported length.
    static const unsigned int MAX_LENGTH = (2 * MAX_RADIUS + 1) * (2 * MAX_RADIUS + 1);

    /// @copydoc IConvolutionFilter::setKernel(const float*, unsigned int)
    void setKernel( const float* kernel, unsigned int length );
};

/**
 * @brief Applies 2D discrete convolution to the image.
 *
 * @author Jan Bobek
 */
class CudaConvolutionFilter
: public ICudaConvolutionFilter
{
protected:
    /// @copydoc ICudaConvolutionFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

/**
 * @brief Applies 1D discrete row convolution to the image.
 *
 * @author Jan Bobek
 */
class CudaRowConvolutionFilter
: public ICudaConvolutionFilter
{
protected:
    /// @copydoc ICudaConvolutionFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

/**
 * @brief Applies 1D discrete column convolution to the image.
 *
 * @author Jan Bobek
 */
class CudaColumnConvolutionFilter
: public ICudaConvolutionFilter
{
protected:
    /// @copydoc ICudaConvolutionFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_CONVOLUTION_FILTER_HXX__INCL__ */
