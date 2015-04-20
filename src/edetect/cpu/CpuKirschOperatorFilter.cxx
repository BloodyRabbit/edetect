/** @file
 * @brief Definition of CpuKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuKirschOperatorFilter.hxx"

/*************************************************************************/
/* CpuKirschOperatorFilter                                               */
/*************************************************************************/
void
CpuKirschOperatorFilter::applyKirschOperator(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
        for( unsigned int col = 0; col < src.columns(); ++col )
        {
            float* const dstp =
                (float*)(dest.data() + row * dest.stride()) + col;
            const unsigned char* const srcp =
                src.data() + row * src.stride() + col * sizeof(float);

            const float* const tp =
                (const float*)(0 < row ? srcp - src.stride() : srcp);
            const float* const mp =
                (const float*)srcp;
            const float* const bp =
                (const float*)(row + 1 < src.rows() ? srcp + src.stride() : srcp);

            const int li = (0 < col ? -1 : 0);
            const int ri = (col + 1 < src.columns() ? 1 : 0);

            float x =
                5.0f * (tp[li] + tp[ 0] + tp[ri]) -
                3.0f * (mp[li] + mp[ri] + bp[li] + bp[0] + bp[ri]);

            float a = fabs(x);
            a = fmaxf( a, fabs( x += 8.0f * (mp[ri] - tp[li]) ) );
            a = fmaxf( a, fabs( x += 8.0f * (bp[ri] - tp[ 0]) ) );
            a = fmaxf( a, fabs( x += 8.0f * (bp[ 0] - tp[ri]) ) );
            a = fmaxf( a, fabs( x += 8.0f * (bp[li] - mp[ri]) ) );
            a = fmaxf( a, fabs( x += 8.0f * (mp[li] - bp[ri]) ) );
            a = fmaxf( a, fabs( x += 8.0f * (tp[li] - bp[ 0]) ) );
            a = fmaxf( a, fabs( x += 8.0f * (tp[ 0] - bp[li]) ) );

            *dstp = a;
        }
}
