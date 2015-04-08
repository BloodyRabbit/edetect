/** @file
 * @brief Declaration of CudaImage class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_IMAGE_HXX__INCL__
#define CUDA__CUDA_IMAGE_HXX__INCL__

#include "MemImage.hxx"

/**
 * @brief An image stored at CUDA device.
 *
 * @author Jan Bobek
 */
class CudaImage
: public MemImage
{
public:
    /**
     * @brief Deallocates the image.
     */
    ~CudaImage();

    /// @copydoc IImage::load(const void*, unsigned int, unsigned int, unsigned int, Image::Format)
    void load(
        const void* data,
        unsigned int rows,
        unsigned int cols,
        unsigned int stride,
        Image::Format fmt
        );
    /// @copydoc IImage::save(void*, unsigned int)
    void save( void* data, unsigned int stride = 0 ) const;
    /// @copydoc IImage::reset(unsigned int, unsigned int, Image::Format)
    void reset(
        unsigned int rows = 0,
        unsigned int cols = 0,
        Image::Format fmt = Image::FMT_INVALID
        );

    // Eliminates annoying warnings
    using IImage::swap;
    /// @copydoc IImage::swap(IImage&)
    void swap( IImage& oth );
    /// @coppydoc IImage::swap(CudaImage&)
    void swap( CudaImage& oth );

    // Eliminates annoying warnings
    using IImage::operator=;
    /// @copydoc IImage::operator=(const CudaImage&)
    CudaImage& operator=( const CudaImage& oth );

protected:
    /// @copydoc IImage::apply(IImageFilter&)
    void apply( IImageFilter& filter );
    /// @copydoc IImage::duplicate(IImage&)
    void duplicate( IImage& dest ) const;
};

#endif /* !CUDA__CUDA_IMAGE_HXX__INCL__ */
