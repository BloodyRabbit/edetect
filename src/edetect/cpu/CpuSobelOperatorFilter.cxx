/** @file
 * @brief Definition of CpuSobelOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cpu/CpuSobelOperatorFilter.hxx"

/*************************************************************************/
/* CpuSobelOperatorFilter                                                */
/*************************************************************************/
void
CpuSobelOperatorFilter::computeGradient(
    IImage& vert,
    const IImage& horz
    )
{
    for( unsigned int row = 0; row < vert.rows(); ++row )
    {
        float* const vertp =
            (float*)(vert.data() + row * vert.stride());
        const float* const horzp =
            (const float*)(horz.data() + row * horz.stride());

        for( unsigned int col = 0; col < vert.columns(); ++col )
            vertp[col] = std::sqrt( vertp[col] * vertp[col] + horzp[col] * horzp[col] );
    }
}
