/** @file
 * @brief Definition of class CpuMultiplyFilter.
 *
 * @author Jan Bobek
 * @since 19th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuMultiplyFilter.hxx"

/*************************************************************************/
/* CpuMultiplyFilter                                                     */
/*************************************************************************/
CpuMultiplyFilter::CpuMultiplyFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: IMultiplyFilter( first, second )
{
}

void
CpuMultiplyFilter::filter2(
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
            dstp[col] *= srcp[col];
    }
}
