/** @file
 * @brief Definition of CudaFilterPipeline class.
 *
 * @author Jan Bobek
 */

#include "common.hxx"
#include "CudaFilterPipeline.hxx"

/*************************************************************************/
/* CudaFilterPipeline                                                    */
/*************************************************************************/
CudaFilterPipeline::~CudaFilterPipeline()
{
    clear();
}

void
CudaFilterPipeline::add(
    CudaFilter* filter
    )
{
    mFilters.push_back( filter );
}

void
CudaFilterPipeline::clear()
{
    std::list< CudaFilter* >::iterator cur, end;
    cur = mFilters.begin();
    end = mFilters.end();
    while( cur != end )
        delete *cur, cur = mFilters.erase( cur );
}

void
CudaFilterPipeline::process(
    CudaImage& image
    )
{
    std::list< CudaFilter* >::const_iterator cur, end;
    cur = mFilters.begin();
    end = mFilters.end();
    for(; cur != end; ++cur )
        (*cur)->process( image );
}
