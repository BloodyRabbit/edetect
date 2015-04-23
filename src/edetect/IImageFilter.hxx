/** @file
 * @brief Declaration of class IImageFilter.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#ifndef IIMAGE_FILTER_HXX__INCL__
#define IIMAGE_FILTER_HXX__INCL__

class IImage;

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
    virtual void filter( IImage& image ) = 0;
    /// @copydoc ImageFilter::setParam(const char*, ...)
    void setParam( const char* name, ... );
    /// @copydoc ImageFilter::setParamVa(const char*, va_list)
    virtual void setParamVa( const char* name, va_list ap );
};

#include "IImageFilter.ixx"

#endif /* !IIMAGE_FILTER_HXX__INCL__ */
