/** @file
 * @brief Definition of class CpuInvertFilter.
 *
 * @author Jan Bobek
 * @since 25th April 2015
 */

#include "edetect.hxx"
#include "IImage.hxx"
#include "cpu/CpuInvertFilter.hxx"

/*************************************************************************/
/* CpuInvertFilter                                                       */
/*************************************************************************/
void
CpuInvertFilter::invert(
    IImage& image
    )
{
    for( unsigned int row = 0; row < image.rows(); ++row )
    {
        float* const rowp =
            (float*)(image.data() + row * image.stride());

        for( unsigned int col = 0; col < image.columns(); ++col )
            rowp[col] = 1.0f - rowp[col];
    }
}
