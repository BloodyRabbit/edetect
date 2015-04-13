/** @file
 * @brief Definition of CpuMarrHildrethOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuMarrHildrethOperatorFilter.hxx"

/*************************************************************************/
/* CpuMarrHildrethOperatorFilter                                         */
/*************************************************************************/
void
CpuMarrHildrethOperatorFilter::mergeEdges(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        float* const dstp =
            (float*)(dest.data() + row * dest.stride());
        const float* const srcp =
            (float*)(src.data() + row * src.stride());

        for( unsigned int col = 0; col < src.columns(); ++col )
            dstp[col] *= srcp[col];
    }
}
