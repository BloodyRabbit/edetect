/** @file
 * @brief Definition of class ImageBackend.
 *
 * @author Jan Bobek
 * @since 7th April 2015
 */

#include "edetect.hxx"
#include "ImageBackend.hxx"

#include "IImageBackend.hxx"
#include "cpu/CpuBackend.hxx"
#include "cuda/CudaBackend.hxx"

/*************************************************************************/
/* ImageBackend                                                          */
/*************************************************************************/
ImageBackend::ImageBackend(
    const char* name
    )
: mBackend( NULL )
{
    if( !strcmp( name, "cpu" ) )
        mBackend = new CpuBackend;
    else if( !strcmp( name, "cuda" ) )
        mBackend = new CudaBackend;
    else
        throw std::invalid_argument(
            "ImageBackend: Backend not implemented" );
}

ImageBackend::~ImageBackend()
{
    delete mBackend;
}

IImage*
ImageBackend::createImage()
{
    return mBackend->createImage();
}

IImageFilter*
ImageBackend::createFilter(
    const char* name
    )
{
    return mBackend->createFilter( name );
}
