/** @file
 * @brief Declaration of class IGeneratedKernelFilter.
 *
 * @author Jan Bobek
 * @since 16th April 2015
 */

#ifndef IGENERATED_KERNEL_FILTER_HXX__INCL__
#define IGENERATED_KERNEL_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a filter which generates
 *   its convolution kernel on-the-fly.
 *
 * @author Jan Bobek
 */
template< typename CF >
class IGeneratedKernelFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    IGeneratedKernelFilter();
    /**
     * @brief Frees the generated kernel.
     */
    ~IGeneratedKernelFilter();

    /**
     * @brief Applies the generated kernel to the image.
     *
     * @param[in,out] image
     *   The image to apply the blur to.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

    /**
     * @brief Sets radius of the generated kernel.
     */
    void setRadius( unsigned int radius );

protected:
    /**
     * @brief Generates the kernel.
     *
     * @param[in] radius
     *   Radius of the kernel to generate.
     *
     * @return
     *   The generated kernel.
     */
    virtual float* generateKernel( unsigned int radius ) = 0;

    /// The generated kernel.
    float* mKernel;
    /// The convolution filter we are delegating to.
    CF mFilter;
};

#include "IGeneratedKernelFilter.txx"

#endif /* !IGENERATED_KERNEL_FILTER_HXX__INCL__ */
