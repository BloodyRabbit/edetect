/** @file
 * @brief Definition of class CpuIntFloatFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "edetect.hxx"
#include "CpuIntFloatFilter.hxx"

/*************************************************************************/
/* CpuIntFloatFilter                                                     */
/*************************************************************************/
void
CpuIntFloatFilter::convertInt2Float(
    IImage& dest, const IImage& src
    )
{
    const unsigned int columns =
        src.columns() * Image::channels( src.format() );

    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        float* const dstp =
            (float*)(dest.data() + row * dest.stride());
        const unsigned char* const srcp =
            src.data() + row * src.stride();

        for( unsigned int col = 0; col < columns; ++col )
            dstp[col] = srcp[col] / 255.0f;
    }
}

void
CpuIntFloatFilter::convertFloat2Int(
    IImage& dest, const IImage& src
    )
{
    const unsigned int columns =
        src.columns() * Image::channels( src.format() );

    for( unsigned int row = 0; row < src.rows(); ++row )
    {
        unsigned char* const dstp =
            dest.data() + row * dest.stride();
        const float* const srcp =
            (float*)(src.data() + row * src.stride());

        for( unsigned int col = 0; col < columns; ++col )
            dstp[col] = (unsigned char)(
                255.0f * std::max( 0.0f, std::min( srcp[col], 1.0f ) )
                );
    }
}
