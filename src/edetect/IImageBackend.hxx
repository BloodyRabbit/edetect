/** @file
 * @brief Declaration of class IImageBackend.
 *
 * @author Jan Bobek
 * @since 2nd April 2015
 */

#ifndef IIMAGE_BACKEND_HXX__INCL__
#define IIMAGE_BACKEND_HXX__INCL__

class IImage;
class IImageFilter;

/**
 * @brief Interface of an image backend.
 *
 * @author Jan Bobek
 */
class IImageBackend
{
public:
    /**
     * @brief Releases the backend resources.
     */
    virtual ~IImageBackend() {}

    /// @copydoc ImageBackend::createImage()
    virtual IImage* createImage() = 0;
    /// @copydoc ImageBackend::createFilter(const char*)
    virtual IImageFilter* createFilter( const char* name ) = 0;
};

#endif /* !IIMAGE_BACKEND_HXX__INCL__ */
