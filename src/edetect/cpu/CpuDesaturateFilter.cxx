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
CpuDesaturateFilter::desaturateAverageInt(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        unsigned char* const dstp =
            dest.data() + row * dest.stride();
        const unsigned char* const srcp =
            src.data() + row * src.stride();

        for( unsigned int col = 0; col < src.columns(); ++col )
            dstp[col] = ((unsigned int)
                         srcp[3 * col + 0] +
                         srcp[3 * col + 1] +
                         srcp[3 * col + 2] + 1) / 3;
    }
}

void
CpuDesaturateFilter::desaturateAverageFloat(
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
            dstp[col] = (srcp[3 * col + 0] +
                         srcp[3 * col + 1] +
                         srcp[3 * col + 2]) / 3.0f;
    }
}

void
CpuDesaturateFilter::desaturateLightnessInt(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        unsigned char* const dstp =
            dest.data() + row * dest.stride();
        const unsigned char* const srcp =
            src.data() + row * src.stride();

        for( unsigned int col = 0; col < src.columns(); ++col )
        {
            const unsigned char
                a = std::min( srcp[3 * col + 0], srcp[3 * col + 1] ),
                b = std::max( srcp[3 * col + 0], srcp[3 * col + 1] );
            const unsigned int
                c = std::min( srcp[3 * col + 2], a ),
                d = std::max( srcp[3 * col + 2], b );

            dstp[col] = (c + d + 1) / 2;
        }
    }
}

void
CpuDesaturateFilter::desaturateLightnessFloat(
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
            const float
                a = std::min( srcp[3 * col + 0], srcp[3 * col + 1] ),
                b = std::max( srcp[3 * col + 0], srcp[3 * col + 1] ),
                c = std::min( srcp[3 * col + 2], a ),
                d = std::max( srcp[3 * col + 2], b );

            dstp[col] = 0.5f * (c + d);
        }
    }
}

void
CpuDesaturateFilter::desaturateLuminosityInt(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        unsigned char* const dstp =
            dest.data() + row * dest.stride();
        const unsigned char* const srcp =
            src.data() + row * src.stride();

        for( unsigned int col = 0; col < src.columns(); ++col )
            /* +2:RED +1:GREEN +0:BLUE */
            dstp[col] = (0.2126f * srcp[3 * col + 2] +
                         0.7152f * srcp[3 * col + 1] +
                         0.0722f * srcp[3 * col + 0]
                         + 0.5f);
    }
}

void
CpuDesaturateFilter::desaturateLuminosityFloat(
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
            /* +2:RED +1:GREEN +0:BLUE */
            dstp[col] = (0.2126f * srcp[3 * col + 2] +
                         0.7152f * srcp[3 * col + 1] +
                         0.0722f * srcp[3 * col + 0]);
    }
}
