/** @file
 * @brief Definition of class CpuConvolution2dSeparableFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuConvolution2dSeparableFilter.hxx"

/*************************************************************************/
/* CpuConvolution2dSeparableFilter                                       */
/*************************************************************************/
void
CpuConvolution2dSeparableFilter::convolveRows(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
        for( unsigned int col = 0; col < src.columns(); ++col )
        {
            const unsigned int start =
                (col < mRowKernelRadius ? mRowKernelRadius - col : 0);
            const unsigned int end =
                (src.columns() <= col + mRowKernelRadius
                 ? src.columns() - col + mRowKernelRadius - 1
                 : 2 * mRowKernelRadius);

            float* dstp =
                (float*)(dest.data() + row * dest.stride())
                + col;
            const float* srcp =
                (const float*)(src.data() + row * src.stride())
                + (col - mRowKernelRadius + start);

            unsigned int k;
            float x = 0.0f;

            for( k = 0; k < start; ++k )
                x += *srcp * mRowKernel[k];
            for(; k < end; ++k, ++srcp )
                x += *srcp * mRowKernel[k];
            for(; k <= 2 * mRowKernelRadius; ++k )
                x += *srcp * mRowKernel[k];

            *dstp = x;
        }
}

void
CpuConvolution2dSeparableFilter::convolveColumns(
    IImage& dest,
    const IImage& src
    )
{
    for( unsigned int row = 0; row < src.rows(); ++row )
        for( unsigned int col = 0; col < src.columns(); ++col )
        {
            const unsigned int start =
                (row < mColumnKernelRadius ? mColumnKernelRadius - row : 0);
            const unsigned int end =
                (src.rows() <= row + mColumnKernelRadius
                 ? src.rows() - row + mColumnKernelRadius - 1
                 : 2 * mColumnKernelRadius);

            float* dstp =
                (float*)(dest.data() + row * dest.stride())
                + col;
            const unsigned char* srcp = src.data()
                + (row - mColumnKernelRadius + start) * src.stride()
                + col * sizeof(float);

            unsigned int i;
            float x = 0.0f;

            for( i = 0; i < start; ++i )
                x += *(float*)srcp * mColumnKernel[i];
            for(; i < end; ++i, srcp += src.stride() )
                x += *(float*)srcp * mColumnKernel[i];
            for(; i <= 2 * mColumnKernelRadius; ++i )
                x += *(float*)srcp * mColumnKernel[i];

            *dstp = x;
        }
}
