/** @file
 * @brief Definition of ImageFilterPipeline class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "ImageFilterPipeline.hxx"

/*************************************************************************/
/* ImageFilterPipeline                                                   */
/*************************************************************************/
ImageFilterPipeline::~ImageFilterPipeline()
{
    clear();
}

void
ImageFilterPipeline::add(
    IImageFilter* filter
    )
{
    mFilters.push_back( filter );
}

void
ImageFilterPipeline::clear()
{
    std::list< IImageFilter* >::iterator cur, end;
    cur = mFilters.begin();
    end = mFilters.end();
    while( cur != end )
        delete *cur, cur = mFilters.erase( cur );
}

void
ImageFilterPipeline::filter(
    IImage& image
    )
{
    std::list< IImageFilter* >::const_iterator cur, end;
    cur = mFilters.begin();
    end = mFilters.end();
    for(; cur != end; ++cur )
        (*cur)->filter( image );
}
