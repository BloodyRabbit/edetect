/** @file
 * @brief Declaration of CudaImage class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA_IMAGE_HXX__INCL__
#define CUDA_IMAGE_HXX__INCL__

/**
 * @brief An image stored at CUDA device.
 *
 * @author Jan Bobek
 */
class CudaImage
{
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
    static size_t channels( Format fmt ) { return FMT_CHANNELS[fmt]; }
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
    static size_t channelSize( Format fmt ) { return FMT_CHANNEL_SIZE[fmt]; }
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
    static size_t pixelSize( Format fmt ) { return channels( fmt ) * channelSize( fmt ); }

    /**
     * @brief Initializes an empty image.
     *
     * @param[in] rows
     *   Number of rows in the image.
     * @param[in] cols
     *   Number of columns in the image.
     * @param[in] fmt
     *   Format of the image.
     *
     * @throw CudaError
     *   Not enough memory on device.
     */
    CudaImage(
        size_t rows = 0,
        size_t cols = 0,
        Format fmt = FMT_INVALID
        );
    /**
     * @brief Initializes the image.
     *
     * @param[in] file
     *   Name of the file to load image from.
     *
     * @throw std::runtime_error
     *   Failed to open the file or load the image.
     * @throw CudaError
     *   Not enough memory on device.
     */
    CudaImage( const char* file );
    /**
     * @brief Initializes the image.
     *
     * @param[in] img
     *   The image data.
     * @param[in] rows
     *   Number of rows in the image.
     * @param[in] cols
     *   Number of columns in the image.
     * @param[in] rowStride
     *   Size of the row stride in bytes.
     * @param[in] fmt
     *   Format of the image.
     *
     * @throw CudaError
     *   Not enough memory on device.
     */
    CudaImage(
        const void* img,
        size_t rows,
        size_t cols,
        size_t rowStride,
        Format fmt
        );
    /**
     * @brief Duplicates an image.
     *
     * @param[in] oth
     *   The image to duplicate.
     */
    CudaImage( const CudaImage& oth );
    /**
     * @brief Releases resources held by the image object.
     */
    ~CudaImage();

    /**
     * @brief Obtains pointer to the image data.
     *
     * @return
     *   Pointer to the image data.
     */
    void* data() { return mImage; }
    /**
     * @brief Obtains pointer to the image data.
     *
     * @return
     *   Pointer to the image data.
     */
    const void* data() const { return mImage; }

    /**
     * @brief Obtains number of rows in the image.
     *
     * @return
     *   Number of rows in the image.
     */
    size_t rows() const { return mRows; }
    /**
     * @brief Obtains number of columns in the image.
     *
     * @return
     *   Number of columns in the image.
     */
    size_t columns() const { return mColumns; }
    /**
     * @brief Obtains size of the row stride (in bytes).
     *
     * @return
     *   Size of the row stride (in bytes).
     */
    size_t rowStride() const { return mRowStride; }

    /**
     * @brief Obtains format of the image.
     *
     * @return
     *   Format of the image.
     */
    Format format() const { return mFmt; }
    /**
     * @brief Obtains number of channels in a single pixel.
     *
     * @return
     *   Number of channels in a single pixel.
     */
    size_t channels() const { return channels( mFmt ); }
    /**
     * @brief Obtains size of a single pixel channel.
     *
     * @return
     *   Size of a single pixel channel.
     */
    size_t channelSize() const { return channelSize( mFmt ); }
    /**
     * @brief Obtains size of a single pixel.
     *
     * @return
     *   Size of a single pixel.
     */
    size_t pixelSize() const { return pixelSize( mFmt ); }

    /**
     * @brief Loads the image from a file.
     *
     * @param[in] file
     *   Name of the file.
     *
     * @throw std::runtime_error
     *   Failed to open the file or load the image.
     * @throw CudaError
     *   Not enough memory on device.
     */
    void load( const char* file );
    /**
     * @brief Loads the image from memory.
     *
     * @param[in] img
     *   The image data.
     * @param[in] rows
     *   Number of rows in the image.
     * @param[in] cols
     *   Number of columns in the image.
     * @param[in] rowStride
     *   Size of the row stride (in bytes).
     * @param[in] fmt
     *   Format of the image.
     *
     * @throw CudaError
     *   Not enough memory on device.
     */
    void load(
        const void* img,
        size_t rows,
        size_t cols,
        size_t rowStride,
        Format fmt
        );

    /**
     * @brief Saves the image to a file.
     *
     * @param[in] file
     *   Name of the file.
     *
     * @throw std::runtime_error
     *   Failed to save the image.
     * @throw CudaError
     *   Not enough memory on device.
     */
    void save( const char* file );
    /**
     * @brief Saves the image to memory.
     *
     * @param[out] img
     *   Where to store the image data.
     * @param[in,opt] rowStride
     *   The row stride to use (in bytes).
     */
    void save( void* img, size_t rowStride = 0 );

    /**
     * @brief Swaps content of two images.
     *
     * @param[in] oth
     *   The image to swap content with.
     */
    void swap( CudaImage& oth );
    /**
     * @brief Resets the image.
     *
     * @param[in,opt] rows
     *   New number of rows.
     * @param[in,opt] columns
     *   New number of columns.
     * @param[in,opt] fmt
     *   New format.
     *
     * @throw CudaError
     *   Not enough memory on device.
     */
    void reset(
        size_t rows = 0,
        size_t cols = 0,
        Format fmt = FMT_INVALID
        );

    /**
     * @brief Duplicates an image.
     *
     * @param[in] oth
     *   The image to duplicate.
     */
    CudaImage& operator=( const CudaImage& oth );

protected:
    /// The image data.
    void* mImage;
    /// Number of rows in the image.
    size_t mRows;
    /// Number of columns in the image.
    size_t mColumns;
    /// Size of the row stride (in bytes).
    size_t mRowStride;

    /// Format of the image.
    Format mFmt;

    /// Keeps number of channels of each format.
    static const size_t FMT_CHANNELS[];
    /// Keeps channel size (in bytes) of each format.
    static const size_t FMT_CHANNEL_SIZE[];
};

#endif /* !CUDA_IMAGE_HXX__INCL__ */
