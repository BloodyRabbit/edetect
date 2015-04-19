/** @file
 * @brief Definition of class ImageFilterBuilder.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "ImageBackend.hxx"
#include "ImageFilterBuilder.hxx"

#include "IImageFilterBuilder.hxx"
#include "StringFilterBuilder.hxx"

/*************************************************************************/
/* ImageFilterBuilder                                                    */
/*************************************************************************/
ImageFilterBuilder::ImageFilterBuilder(
    char* str
    )
: mBuilder( new StringFilterBuilder( str ) )
{
}

ImageFilterBuilder::~ImageFilterBuilder()
{
    delete mBuilder;
}

IImageFilter*
ImageFilterBuilder::buildFilter(
    ImageBackend& backend
    )
{
    return mBuilder->buildFilter( *backend.mBackend );
}
