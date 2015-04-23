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
class SeparableConvolutionFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     *
     * @param[in] rowFilter
     *   The row convolution filter to use.
     * @param[in] columnFilter
     *   The column convolution filter to use.
     */
    SeparableConvolutionFilter(
        IImageFilter* rowFilter,
        IImageFilter* columnFilter
        );
    /**
     * @brief Releases the filters.
     */
    ~SeparableConvolutionFilter();

    /**
     * @brief Performs convolution on the image.
     *
     * @param[in] image
     *   The image to convolve.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParamVa(const char*, va_list ap)
    void setParamVa( const char* name, va_list ap );

protected:
    /// The row convolution filter.
    IImageFilter* mRowFilter;
    /// The column convolution filter.
    IImageFilter* mColumnFilter;
};

#endif /* !SEPARABLE_CONVOLUTION_FILTER_HXX__INCL__ */
