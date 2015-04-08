/** @file
 * @brief Declaration of class IConvolution2dFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef ICONVOLUTION_2D_FILTER_HXX__INCL__
#define ICONVOLUTION_2D_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a 2D discrete convolution filter.
 *
 * @author Jan Bobek
 */
class IConvolution2dFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    IConvolution2dFilter();

    /**
     * @brief Performs convolution on the image.
     *
     * @param[in] image
     *   The image to convolve.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

    /**
     * @brief Sets a new convolution kernel.
     *
     * @param[in] kernel
     *   The new kernel.
     * @param[in] radius
     *   Radius of the new kernel.
     */
    virtual void setKernel( const float* kernel, unsigned int radius );

protected:
    /**
     * @brief Performs 2D discrete convolution.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void convolve( IImage& dest, const IImage& src ) = 0;

    /// The convolution kernel.
    const float* mKernel;
    /// Radius of the kernel.
    unsigned int mRadius;
};

#endif /* !ICONVOLUTION_2D_FILTER_HXX__INCL__ */
