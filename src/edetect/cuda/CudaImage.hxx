/** @file
 * @brief Declaration of CudaImage class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_IMAGE_HXX__INCL__
#define CUDA__CUDA_IMAGE_HXX__INCL__

#include "IImage.hxx"

/**
 * @brief An image stored at CUDA device.
 *
 * @author Jan Bobek
 */
class CudaImage
: public IImage
{
public:
    /**
     * @brief Initializes the image.
     */
    CudaImage();
    /**
     * @brief Deallocates the image.
     */
    ~CudaImage();

    /// @copydoc IImage::data()
    unsigned char* data();
    /// @copydoc IImage::data() const
    const unsigned char* data() const;

    /// @copydoc IImage::rows() const
    unsigned int rows() const;
    /// @copydoc IImage::columns() const
    unsigned int columns() const;
    /// @copydoc IImage::stride() const
    unsigned int stride() const;

    /// @copydoc IImage::format() const
    Image::Format format() const;

    /// @copydoc IImage::load(const char*)
    void load( const char* file );
    /// @copydoc IImage::load(const void*, unsigned int, unsigned int, unsigned int, Image::Format)
    void load(
        const void* data,
        unsigned int rows,
        unsigned int cols,
        unsigned int stride,
        Image::Format fmt
        );

    /// @copydoc IImage::save(const char*)
    void save( const char* file );
    /// @copydoc IImage::save(void*, unsigned int)
    void save( void* data, unsigned int stride = 0 );

    /// @copydoc IImage::reset(unsigned int, unsigned int, Image::Format)
    void reset(
        unsigned int rows = 0,
        unsigned int cols = 0,
        Image::Format fmt = Image::FMT_INVALID
        );

    /// @copydoc IImage::swap(IImage&)
    void swap( IImage& oth );
    /// @coppydoc IImage::swap(CudaImage&)
    void swap( CudaImage& oth );

    /// @copydoc IImage::operator=(const CudaImage&)
    CudaImage& operator=( const CudaImage& oth );

protected:
    /// @copydoc IImage::apply(IImageFilter&)
    void apply( IImageFilter& filter );
    /// @copydoc IImage::duplicate(IImage&)
    void duplicate( IImage& dest ) const;

    /// The image data.
    unsigned char* mData;
    /// Number of rows in the image.
    unsigned int mRows;
    /// Number of columns in the image.
    unsigned int mColumns;
    /// Size of the row stride (in bytes).
    unsigned int mStride;

    /// Format of the image.
    Image::Format mFmt;
};

#endif /* !CUDA__CUDA_IMAGE_HXX__INCL__ */
