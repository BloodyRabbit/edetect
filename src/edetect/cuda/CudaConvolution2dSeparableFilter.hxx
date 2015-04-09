/** @file
 * @brief Declaration of CudaConvolution2dSeparableFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__
#define CUDA__CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__

#include "IConvolution2dSeparableFilter.hxx"

/**
 * @brief Performs two-dimensional discrete convolution
 *   with separable kernel on the image.
 *
 * @author Jan Bobek
 */
class CudaConvolution2dSeparableFilter
: public IConvolution2dSeparableFilter
{
public:
    /// The largest supported radius.
    static const unsigned int MAX_RADIUS = 32;

    /// @copydoc IConvolution2dSeparableFilter::setRowKernel(const float*, unsigned int)
    void setRowKernel( const float* kernel, unsigned int radius );
    /// @copydoc IConvolution2dSeparableFilter::setColumnKernel(const float*, unsigned int)
    void setColumnKernel( const float* kernel, unsigned int radius );

protected:
    /// @copydoc IConvolution2dSeparableFilter::convolveRows(IImage&, const IImage&)
    void convolveRows( IImage& dest, const IImage& src );
    /// @copydoc IConvolution2dSeparableFilter::convolveColumns(IImage&, const IImage&)
    void convolveColumns( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__ */
