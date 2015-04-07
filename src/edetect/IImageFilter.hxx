/** @file
 * @brief Declaration of class IImageFilter.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#ifndef IIMAGE_FILTER_HXX__INCL__
#define IIMAGE_FILTER_HXX__INCL__

class IImage;
class CudaImage;

/**
 * @brief Interface of an image filter.
 *
 * @author Jan Bobek
 */
class IImageFilter
{
public:
    /**
     * @brief Destroys the filter.
     */
    virtual ~IImageFilter();

    /// @copydoc ImageFilter::filter(Image&)
    void filter( IImage& image );
    /**
     * @brief Applies the filter to a CUDA image.
     *
     * @param[in,out] image
     *   The image to filter.
     */
    virtual void filter( CudaImage& image );

    /// @copydoc ImageFilter::setParam(const char*, const void*)
    virtual void setParam( const char* name, const void* value );
};

#include "IImageFilter.ixx"

#endif /* !IIMAGE_FILTER_HXX__INCL__ */
