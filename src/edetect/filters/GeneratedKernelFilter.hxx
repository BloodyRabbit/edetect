/** @file
 * @brief Declaration of class GeneratedKernelFilter.
 *
 * @author Jan Bobek
 * @since 16th April 2015
 */

#ifndef FILTERS__GENERATED_KERNEL_FILTER_HXX__INCL__
#define FILTERS__GENERATED_KERNEL_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a filter which generates
 *   its convolution kernel on-the-fly.
 *
 * @author Jan Bobek
 */
template< typename K >
class GeneratedKernelFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     *
     * @param[in] filter
     *   The convolution filter to use.
     * @param[in,opt] kernel
     *   The kernel to use.
     */
    GeneratedKernelFilter(
        IImageFilter* filter,
        K kernel = K()
        );
    /**
     * @brief Releases the filter.
     */
    ~GeneratedKernelFilter();

    /**
     * @brief Applies the generated kernel to the image.
     *
     * @param[in,out] image
     *   The image to apply the blur to.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParamVa(const char*, va_list ap)
    void setParamVa( const char* name, va_list ap );

    /**
     * @brief Sets radius of the generated kernel.
     */
    void setRadius( unsigned int radius );

protected:
    /// The convolution filter we are delegating to.
    IImageFilter* mFilter;
    /// The kernel generator.
    K mKernel;
};

#include "filters/GeneratedKernelFilter.txx"

#endif /* !FILTERS__GENERATED_KERNEL_FILTER_HXX__INCL__ */
