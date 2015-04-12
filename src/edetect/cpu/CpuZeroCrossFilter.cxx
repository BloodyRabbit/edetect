/** @file
 * @brief Definition of class CpuZeroCrossFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuZeroCrossFilter.hxx"

/*************************************************************************/
/* CpuZeroCrossFilter                                                    */
/*************************************************************************/
void
CpuZeroCrossFilter::detectZeroCross(
    IImage& dest,
    const IImage& src
    )
{
    memset( dest.data(),
            0, dest.stride() );
    memset( dest.data() + (dest.rows() - 1) * dest.stride(),
            0, dest.stride() );

    for( unsigned int row = 1; row < (dest.rows() - 1); ++row )
    {
        float* const dstp =
            (float*)(dest.data() + row * dest.stride());
        const unsigned char* const srcp =
            src.data() + row * src.stride();

        dstp[0] = 0.0f;
        dstp[dest.columns() - 1] = 0.0f;

        for( unsigned int col = 1; col < (dest.columns() - 1); ++col )
            dstp[col] =
                // Left-Right
                (0 > (((const float*)(srcp))[col - 1] *
                      ((const float*)(srcp))[col + 1]) ||
                 // Top-Bottom
                 0 > (((const float*)(srcp - src.stride()))[col] *
                      ((const float*)(srcp + src.stride()))[col]) ||
                 // Main diagonal
                 0 > (((const float*)(srcp - src.stride()))[col - 1] *
                      ((const float*)(srcp + src.stride()))[col + 1]) ||
                 // Antidiagonal
                 0 > (((const float*)(srcp - src.stride()))[col + 1] *
                      ((const float*)(srcp + src.stride()))[col - 1])
                 ? 1.0f : 0.0f);
    }
}
