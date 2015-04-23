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
#include "StringFilterBuilderImpl.hxx"
#include "XmlFilterBuilderImpl.hxx"

/*************************************************************************/
/* ImageFilterBuilder                                                    */
/*************************************************************************/
ImageFilterBuilder::ImageFilterBuilder(
    IImageFilterBuilder* builder
    )
: mBuilder( builder )
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

/*************************************************************************/
/* StringFilterBuilder                                                   */
/*************************************************************************/
StringFilterBuilder::StringFilterBuilder(
    char* str
    )
: ImageFilterBuilder( new StringFilterBuilderImpl( str ) )
{
}

/*************************************************************************/
/* XmlFilterBuilder                                                      */
/*************************************************************************/
XmlFilterBuilder::XmlFilterBuilder(
    const char* filename
    )
: ImageFilterBuilder( new XmlFilterBuilderImpl( filename ) )
{
}
