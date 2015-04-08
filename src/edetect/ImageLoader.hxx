/** @file
 * @brief Declaration of class ImageLoader.
 *
 * @author Jan Bobek
 * @since 8th April 2015
 */

#ifndef IMAGE_LOADER_HXX__INCL__
#define IMAGE_LOADER_HXX__INCL__

class IImage;

/**
 * @brief Loads images from files.
 *
 * @author Jan Bobek
 */
class ImageLoader
{
public:
    /**
     * @brief Loads image from a file.
     *
     * @param[out] dest
     *   Where to place the image.
     * @param[in] file
     *   Name of the file to load from.
     */
    static void load( IImage& dest, const char* file );
    /**
     * @brief Saves image to a file.
     *
     * @param[in] file
     *   Name of the destination file.
     * @param[in] image
     *   The image to save.
     */
    static void save( const char* file, const IImage& image );
};

#endif /* !IMAGE_LOADER_HXX__INCL__ */
