/** @file
 * @brief Declaration of class SeparableConvolutionFilter.
 *
 * @author Jan Bobek
 * @since 16th April 2015
 */

#ifndef SEPARABLE_CONVOLUTION_FILTER_HXX__INCL__
#define SEPARABLE_CONVOLUTION_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Applies discrete convolution with
 *   a separable kernel.
 *
 * @author Jan Bobek
 */
template< typename RCF, typename CCF >
class SeparableConvolutionFilter
: public IImageFilter
{
public:
    /**
     * @brief Performs convolution on the image.
     *
     * @param[in] image
     *   The image to convolve.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParamVa(const char*, va_list ap)
    void setParamVa( const char* name, va_list ap );

    /**
     * @brief Sets the row kernel.
     *
     * @param[in] kernel
     *   The row kernel.
     * @param[in] radius
     *   Radius of the row kernel.
     */
    void setRowKernel( const float* kernel, unsigned int radius );
    /**
     * @brief Sets the column kernel.
     *
     * @param[in] kernel
     *   The column kernel.
     * @param[in] radius
     *   Radius of the column kernel.
     */
    void setColumnKernel( const float* kernel, unsigned int radius );

protected:
    /// The row convolution filter.
    RCF mRowFilter;
    /// The column convolution filter.
    CCF mColumnFilter;
};

#include "SeparableConvolutionFilter.txx"

#endif /* !SEPARABLE_CONVOLUTION_FILTER_HXX__INCL__ */
