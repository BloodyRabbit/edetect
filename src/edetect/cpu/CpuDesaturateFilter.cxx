/** @file
 * @brief Definition of class CpuDesaturateFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuDesaturateFilter.hxx"

/*************************************************************************/
/* CpuDesaturateFilter                                                   */
/*************************************************************************/
void
CpuDesaturateFilter::desaturateAverage(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        float* const dstp =
            (float*)(dest.data() + row * dest.stride());
        const float* const srcp =
            (const float*)(src.data() + row * src.stride());

        for( unsigned int col = 0; col < src.columns(); ++col )
            dstp[col] = (
                srcp[3 * col + 0] +
                srcp[3 * col + 1] +
                srcp[3 * col + 2]
                ) / 3.0f;
    }
}

void
CpuDesaturateFilter::desaturateLightness(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        float* const dstp =
            (float*)(dest.data() + row * dest.stride());
        const float* const srcp =
            (const float*)(src.data() + row * src.stride());

        for( unsigned int col = 0; col < src.columns(); ++col )
        {
            const float a = std::min( srcp[3 * col + 0], srcp[3 * col + 1] );
            const float b = std::max( srcp[3 * col + 0], srcp[3 * col + 1] );
            const float c = std::min( srcp[3 * col + 2], a );
            const float d = std::max( srcp[3 * col + 2], b );

            dstp[col] = 0.5f * (c + d);
        }
    }
}

void
CpuDesaturateFilter::desaturateLuminosity(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        float* const dstp =
            (float*)(dest.data() + row * dest.stride());
        const float* const srcp =
            (const float*)(src.data() + row * src.stride());

        for( unsigned int col = 0; col < src.columns(); ++col )
            dstp[col] =
                /* +2:RED +1:GREEN +0:BLUE */
                0.2126f * srcp[3 * col + 2] +
                0.7152f * srcp[3 * col + 1] +
                0.0722f * srcp[3 * col + 0];
    }
}
