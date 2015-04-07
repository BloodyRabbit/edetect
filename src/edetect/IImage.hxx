/** @file
 * @brief Declaration of class IImage.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#ifndef IIMAGE_HXX__INCL__
#define IIMAGE_HXX__INCL__

#include "Image.hxx"

class IImageFilter;

class CudaImage;

/**
 * @brief Interface of an image.
 *
 * @author Jan Bobek
 */
class IImage
{
    friend class IImageFilter;

public:
    /**
     * @brief Destroys the image.
     */
    virtual ~IImage();

    /// @copydoc Image::data()
    virtual unsigned char* data() = 0;
    /// @copydoc Image::data() const
    virtual const unsigned char* data() const = 0;

    /// @copydoc Image::rows() const
    virtual unsigned int rows() const = 0;
    /// @copydoc Image::columns() const
    virtual unsigned int columns() const = 0;
    /// @copydoc Image::stride() const
    virtual unsigned int stride() const = 0;
    /// @copydoc Image::format() const
    virtual Image::Format format() const = 0;

    /// @copydoc Image::load(const char*)
    virtual void load( const char* file ) = 0;
    /// @copydoc Image::load(const void*, unsigned int, unsigned int, unsigned int, Image::Format)
    virtual void load(
        const void* data,
        unsigned int rows,
        unsigned int cols,
        unsigned int stride,
        Image::Format fmt
        ) = 0;

    /// @copydoc Image::save(const char*)
    virtual void save( const char* file ) = 0;
    /// @copydoc Image::save(void*, unsigned int)
    virtual void save( void* data, unsigned int stride = 0 ) = 0;

    /// @copydoc Image::reset(unsigned int, unsigned int, Image::Format)
    virtual void reset(
        unsigned int rows = 0,
        unsigned int cols = 0,
        Image::Format fmt = Image::FMT_INVALID
        ) = 0;

    /// @copydoc Image::swap(Image&)
    virtual void swap( IImage& oth ) = 0;
    /**
     * @brief Swaps with a CUDA image.
     *
     * @param[in,out] oth
     *   The image to swap with.
     */
    virtual void swap( CudaImage& oth );

    /// @copydoc Image::operator=(const Image&)
    IImage& operator=( const IImage& oth );
    /**
     * @brief Duplicates a CUDA image.
     *
     * @param[in] oth
     *   The image to duplicate.
     *
     * @return
     *   The duplicated image.
     */
    virtual CudaImage& operator=( const CudaImage& oth );

protected:
    /**
     * @brief Applies given filter to this image.
     *
     * @param[in] filter
     *   The filter to apply.
     */
    virtual void apply( IImageFilter& filter ) = 0;
    /**
     * @brief Duplicates this image.
     *
     * @param[out] dest
     *   Where to place the duplicate.
     */
    virtual void duplicate( IImage& dest ) const = 0;
};

#include "IImage.ixx"

#endif /* !IIMAGE_HXX__INCL__ */
