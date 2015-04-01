/** @file
 * @brief Declaration of class Image.
 *
 * @author Jan Bobek
 * @since 1st April 2015
 */

#ifndef IMAGE_HXX__INCL__
#define IMAGE_HXX__INCL__

class ImageBackend;

class IImage;

/**
 * @brief An image.
 *
 * @author Jan Bobek
 */
class Image
{
    friend class ImageFilter;

public:
    /**
     * @brief An enum representing supported image formats.
     *
     * @author Jan Bobek
     */
    enum Format
    {
        FMT_INVALID = 0,  ///< Invalid format
        FMT_GRAY_UINT8,   ///< Grayscale, 8bpp [0..255]
        FMT_GRAY_FLOAT32, ///< Grayscale, 32bpp [0..1]
        FMT_RGB_UINT8,    ///< RGB, 24bpp [0..255]
        FMT_RGB_FLOAT32   ///< RGB, 96bpp [0..1]
    };

    /**
     * @brief Obtains number of channels of the specified
     *   format.
     *
     * @param[in] fmt
     *   The format to examine.
     *
     * @return
     *   Number of channels of the specified format.
     */
    static unsigned int channels( Format fmt );
    /**
     * @brief Obtains size of a single pixel channel of
     *   the specified format.
     *
     * @param[in] fmt
     *   The format to examine.
     *
     * @return
     *   Size of a single pixel channel of the specified
     *   format.
     */
    static unsigned int channelSize( Format fmt );
    /**
     * @brief Obtains size of a single pixel of the
     *   specified format.
     *
     * @param[in] fmt
     *   The format to examine.
     *
     * @return
     *   Size of a single pixel of the specified format.
     */
    static unsigned int pixelSize( Format fmt );

    /**
     * @brief Initializes an empty image.
     *
     * @param[in] backend
     *   Backend to back the image.
     * @param[in] rows
     *   Number of rows in the image.
     * @param[in] cols
     *   Number of columns in the image.
     * @param[in] fmt
     *   Format of the image.
     */
    Image(
        ImageBackend& backend,
        unsigned int rows = 0,
        unsigned int cols = 0,
        Format fmt = FMT_INVALID
        );
    /**
     * @brief Initializes the image.
     *
     * @param[in] backend
     *   Backend to back the image.
     * @param[in] file
     *   Name of the file to load image from.
     */
    Image( ImageBackend& backend, const char* file );
    /**
     * @brief Initializes the image.
     *
     * @param[in] backend
     *   Backend to back the image.
     * @param[in] data
     *   The image data.
     * @param[in] rows
     *   Number of rows in the image.
     * @param[in] cols
     *   Number of columns in the image.
     * @param[in] stride
     *   Size of the row stride in bytes.
     * @param[in] fmt
     *   Format of the image.
     */
    Image(
        ImageBackend& backend,
        const void* data,
        unsigned int rows,
        unsigned int cols,
        unsigned int stride,
        Format fmt
        );
    /**
     * @brief Duplicates an image.
     *
     * @param[in] oth
     *   The image to duplicate.
     */
    Image( const Image& oth );
    /**
     * @brief Destroys the image.
     */
    ~Image();

    /**
     * @brief Obtains pointer to the image data.
     *
     * @return
     *   Pointer to the image data.
     */
    unsigned char* data();
    /**
     * @brief Obtains pointer to the image data.
     *
     * @return
     *   Pointer to the image data.
     */
    const unsigned char* data() const;

    /**
     * @brief Obtains number of rows in the image.
     *
     * @return
     *   Number of rows in the image.
     */
    unsigned int rows() const;
    /**
     * @brief Obtains number of columns in the image.
     *
     * @return
     *   Number of columns in the image.
     */
    unsigned int columns() const;
    /**
     * @brief Obtains size of the row stride (in bytes).
     *
     * @return
     *   Size of the row stride (in bytes).
     */
    unsigned int stride() const;

    /**
     * @brief Obtains format of the image.
     *
     * @return
     *   Format of the image.
     */
    Format format() const;
    /**
     * @brief Obtains number of channels in a single pixel.
     *
     * @return
     *   Number of channels in a single pixel.
     */
    unsigned int channels() const;
    /**
     * @brief Obtains size of a single pixel channel.
     *
     * @return
     *   Size of a single pixel channel.
     */
    unsigned int channelSize() const;
    /**
     * @brief Obtains size of a single pixel.
     *
     * @return
     *   Size of a single pixel.
     */
    unsigned int pixelSize() const;

    /**
     * @brief Loads the image from a file.
     *
     * @param[in] file
     *   Name of the file.
     */
    void load( const char* file );
    /**
     * @brief Loads the image from memory.
     *
     * @param[in] data
     *   The image data.
     * @param[in] rows
     *   Number of rows in the image.
     * @param[in] cols
     *   Number of columns in the image.
     * @param[in] stride
     *   Size of the row stride (in bytes).
     * @param[in] fmt
     *   Format of the image.
     */
    void load(
        const void* data,
        unsigned int rows,
        unsigned int cols,
        unsigned int stride,
        Format fmt
        );

    /**
     * @brief Saves the image to a file.
     *
     * @param[in] file
     *   Name of the file.
     */
    void save( const char* file );
    /**
     * @brief Saves the image to memory.
     *
     * @param[out] data
     *   Where to store the image data.
     * @param[in,opt] stride
     *   The row stride to use (in bytes).
     */
    void save( void* data, unsigned int stride = 0 );

    /**
     * @brief Resets the image.
     *
     * @param[in,opt] rows
     *   New number of rows.
     * @param[in,opt] cols
     *   New number of columns.
     * @param[in,opt] fmt
     *   New format.
     */
    void reset(
        unsigned int rows = 0,
        unsigned int cols = 0,
        Format fmt = FMT_INVALID
        );

    /**
     * @brief Swaps content of two images.
     *
     * @param[in,out] oth
     *   The image to swap content with.
     */
    void swap( Image& oth );

protected:
    // Disable copy operator
    Image& operator=( const Image& oth );

    /// Implementation of the image.
    IImage* mImage;
};

#endif /* !IMAGE_HXX__INCL__ */
