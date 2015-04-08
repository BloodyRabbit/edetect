/** @file
 * @brief Declaration of class IConvolution2dSeparableFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef ICONVOLUTION_2D_SEPARABLE_FILTER__INCL__
#define ICONVOLUTION_2D_SEPARABLE_FILTER__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a 2D separable convolution filter.
 *
 * @author Jan Bobek
 */
class IConvolution2dSeparableFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    IConvolution2dSeparableFilter();

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
     * @brief Sets the row kernel.
     *
     * @param[in] kernel
     *   The row kernel.
     * @param[in] radius
     *   Radius of the row kernel.
     */
    virtual void setRowKernel(
        const float* kernel,
        unsigned int radius
        );
    /**
     * @brief Sets the column kernel.
     *
     * @param[in] kernel
     *   The column kernel.
     * @param[in] radius
     *   Radius of the column kernel.
     */
    virtual void setColumnKernel(
        const float* kernel,
        unsigned int radius
        );

protected:
    /**
     * @brief Performs one-dimensional convolution along rows.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void convolveRows( IImage& dest, const IImage& src ) = 0;
    /**
     * @brief Performs one-dimensional convolution along columns.
     *
     * @param[out] dest
     *   Where to place the results.
     * @param[in] src
     *   The image data to convolve.
     */
    virtual void convolveColumns( IImage& dest, const IImage& src ) = 0;

    /// The row kernel.
    const float* mRowKernel;
    /// Radius of the row kernel.
    unsigned int mRowKernelRadius;

    /// The column kernel.
    const float* mColumnKernel;
    /// Radius of the column kernel.
    unsigned int mColumnKernelRadius;
};

#endif /* !ICONVOLUTION_2D_SEPARABLE_FILTER__INCL__ */
