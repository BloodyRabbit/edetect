/** @file
 * @brief Definition of ImageFilterPipeline class.
 *
 * @author Jan Bobek
 */

#include "edetect-proc.hxx"
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
    ImageFilter* filter
    )
{
    mFilters.push_back( filter );
}

void
ImageFilterPipeline::clear()
{
    std::list< ImageFilter* >::iterator cur, end;
    cur = mFilters.begin();
    end = mFilters.end();
    while( cur != end )
        delete *cur, cur = mFilters.erase( cur );
}

void
ImageFilterPipeline::filter(
    Image& image
    )
{
    std::list< ImageFilter* >::const_iterator cur, end;
    cur = mFilters.begin();
    end = mFilters.end();
    for(; cur != end; ++cur )
        (*cur)->filter( image );
}
