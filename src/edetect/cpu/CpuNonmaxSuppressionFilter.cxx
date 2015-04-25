/** @file
 * @brief Definition of class CpuNonmaxSuppressionFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuNonmaxSuppressionFilter.hxx"

/*************************************************************************/
/* CpuNonmaxSuppressionFilter                                            */
/*************************************************************************/
CpuNonmaxSuppressionFilter::CpuNonmaxSuppressionFilter(
    IImageFilter* first,
    IImageFilter* second
    )
: INonmaxSuppressionFilter( first, second )
{
}

void
CpuNonmaxSuppressionFilter::nonmaxSuppress(
    IImage& dest,
    const IImage& first,
    const IImage& second
    )
{
    for( unsigned int row = 0; row < dest.rows(); ++row )
        for( unsigned int col = 0; col < dest.columns(); ++col )
        {
            float* const dstp =
                (float*)(dest.data() + row * dest.stride())
                + col;

            const unsigned char* const gxp =
                first.data() + row * first.stride()
                + col * sizeof(float);
            const unsigned char* const gyp =
                second.data() + row * second.stride()
                + col * sizeof(float);

            const float* const gxtp =
                (const float*)(gxp - first.stride());
            const float* const gxmp =
                (const float*)gxp;
            const float* const gxbp =
                (const float*)(gxp + first.stride());

            const float* const gytp =
                (const float*)(gyp - second.stride());
            const float* const gymp =
                (const float*)gyp;
            const float* const gybp =
                (const float*)(gyp + second.stride());

            const float gx = gxmp[0];
            const float gy = gymp[0];
            float gm = std::sqrt( gx * gx + gy * gy );

            if( 0 < row && row < (dest.rows()    - 1) &&
                0 < col && col < (dest.columns() - 1) &&
                0.0f < gm )
            {
                float q, gm1, gm2, gm3, gm4;

                if( 0 < gx * gy )
                    (gm1 = std::sqrt( gxtp[-1] * gxtp[-1] + gytp[-1] * gytp[-1] ),
                     gm3 = std::sqrt( gxbp[ 1] * gxbp[ 1] + gybp[ 1] * gybp[ 1] ));
                else
                    (gm1 = std::sqrt( gxtp[ 1] * gxtp[ 1] + gytp[ 1] * gytp[ 1] ),
                     gm3 = std::sqrt( gxbp[-1] * gxbp[-1] + gybp[-1] * gybp[-1] ));

                if( std::abs(gx) < std::abs(gy) )
                    (q   = std::abs(gx) / std::abs(gy),
                     gm2 = std::sqrt( gxtp[0] * gxtp[0] + gytp[0] * gytp[0] ),
                     gm4 = std::sqrt( gxbp[0] * gxbp[0] + gybp[0] * gybp[0] ));
                else
                    (q   = std::abs(gy) / std::abs(gx),
                     gm2 = std::sqrt( gxmp[ 1] * gxmp[ 1] + gymp[ 1] * gymp[ 1] ),
                     gm4 = std::sqrt( gxmp[-1] * gxmp[-1] + gymp[-1] * gymp[-1] ));

                if( gm < (gm2 + q * (gm1 - gm2)) ||
                    gm < (gm4 + q * (gm3 - gm4)) )
                    gm = 0.0f;
            }

            *dstp = gm;
        }
}
