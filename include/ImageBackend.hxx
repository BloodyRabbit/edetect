/** @file
 * @brief Declaration of class ImageBackend.
 *
 * @author Jan Bobek
 * @since 1st April 2015
 */

#ifndef IMAGE_BACKEND_HXX__INCL__
#define IMAGE_BACKEND_HXX__INCL__

class IImage;
class IImageBackend;
class IImageFilter;

/**
 * @brief Backend of image processing operations.
 *
 * @author Jan Bobek
 */
class ImageBackend
{
public:
    /**
     * @brief Initializes an image backend.
     *
     * @param[in] name
     *   Name of the backend to initialize.
     */
    ImageBackend( const char* name );
    /**
     * @brief Destroys the image backend.
     */
    ~ImageBackend();

    /**
     * @brief Creates an image backed by the backend.
     *
     * @return
     *   The created image.
     */
    IImage* createImage();
    /**
     * @brief Creates a filter backed by the backend.
     *
     * @param[in] name
     *   Name of the filter.
     *
     * @return
     *   The created filter.
     */
    IImageFilter* createFilter( const char* name );

protected:
    /// Implementation of the image backend.
    IImageBackend* mBackend;
};

#endif /* !IMAGE_BACKEND_HXX__INCL__ */
