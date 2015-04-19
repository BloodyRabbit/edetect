/** @file
 * @brief Definition of class CpuEuclideanNormFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuEuclideanNormFilter.hxx"

/*************************************************************************/
/* CpuEuclideanNormFilter                                                */
/*************************************************************************/
CpuEuclideanNormFilter::CpuEuclideanNormFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: IEuclideanNormFilter( first, second )
{
}

void
CpuEuclideanNormFilter::filter2(
    IImage& first,
    const IImage& second
    )
{
    for( unsigned int row = 0; row < first.rows(); ++row )
    {
        float* const dstp =
            (float*)(first.data() + row * first.stride());
        const float* const srcp =
            (const float*)(second.data() + row * second.stride());

        for( unsigned int col = 0; col < first.columns(); ++col )
            dstp[col] = std::sqrt( srcp[col] * srcp[col] + dstp[col] * dstp[col] );
    }
}
