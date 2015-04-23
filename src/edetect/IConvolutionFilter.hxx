/** @file
 * @brief Declaration of class IConvolutionFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef ICONVOLUTION_FILTER_HXX__INCL__
#define ICONVOLUTION_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a discrete convolution filter.
 *
 * @author Jan Bobek
 */
class IConvolutionFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    IConvolutionFilter();
    /**
     * @brief Frees the kernel.
     */
    ~IConvolutionFilter();

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
     * @brief Sets a new convolution kernel.
     *
     * @param[in] kernel
     *   The new kernel.
     * @param[in] length
     *   Length of the new kernel.
     */
    virtual void setKernel( const float* kernel, unsigned int length );

protected:
    /**
     * @brief Performs discrete convolution.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void convolve( IImage& dest, const IImage& src ) = 0;

    /// The convolution kernel.
    const float* mKernel;
    /// Length of the kernel.
    unsigned int mLength;
};

#endif /* !ICONVOLUTION_FILTER_HXX__INCL__ */
