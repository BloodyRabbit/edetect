/** @file
 * @brief Declaration of class IImage.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#ifndef IIMAGE_HXX__INCL__
#define IIMAGE_HXX__INCL__

#include "Image.hxx"

/**
 * @brief Interface of an image.
 *
 * @author Jan Bobek
 */
class IImage
{
public:
    /**
     * @brief Initializes the image.
     */
    IImage();
    /**
     * @brief Destroys the image.
     */
    virtual ~IImage();

    /**
     * @brief Obtains clone of the image.
     *
     * @return
     *   Pointer to identical image.
     */
    virtual IImage* clone() const = 0;
    /**
     * @brief Obtains clone of the image implementation.
     *
     * @return
     *   Pointer to identical implementation of IImage.
     */
    virtual IImage* cloneImpl() const = 0;

    /// @copydoc Image::data()
    unsigned char* data();
    /// @copydoc Image::data() const
    const unsigned char* data() const;

    /// @copydoc Image::rows() const
    unsigned int rows() const;
    /// @copydoc Image::columns() const
    unsigned int columns() const;
    /// @copydoc Image::stride() const
    unsigned int stride() const;
    /// @copydoc Image::format() const
    Image::Format format() const;

    /// @copydoc Image::load(const void*, unsigned int, unsigned int, unsigned int, Image::Format)
    virtual void load(
        const void* data,
        unsigned int rows,
        unsigned int cols,
        unsigned int stride,
        Image::Format fmt
        ) = 0;
    /// @copydoc Image::save(void*, unsigned int)
    virtual void save( void* data, unsigned int stride = 0 ) const = 0;
    /// @copydoc Image::reset(unsigned int, unsigned int, Image::Format)
    virtual void reset(
        unsigned int rows = 0,
        unsigned int cols = 0,
        Image::Format fmt = Image::FMT_INVALID
        ) = 0;

    /// @copydoc Image::swap(Image&)
    void swap( IImage& oth );

protected:
    // Disable copy constructor
    IImage( const IImage& oth );
    // Disable copy operator
    IImage& operator=( const IImage& oth );

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

#include "IImage.ixx"

#endif /* !IIMAGE_HXX__INCL__ */
