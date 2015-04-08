/** @file
 * @brief Definition of CpuKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#include "edetect.hxx"
#include "cpu/CpuKirschOperatorFilter.hxx"

/*************************************************************************/
/* CpuKirschOperatorFilter                                               */
/*************************************************************************/
void
CpuKirschOperatorFilter::computeGradient(
    IImage* images[KERNEL_COUNT]
    )
{
    for( unsigned int row = 0; row < images[0]->rows(); ++row )
    {
        float* srcp[KERNEL_COUNT];
        for( unsigned int i = 0; i < KERNEL_COUNT; ++i )
            srcp[i] = (float*)(images[i]->data() + row * images[i]->stride());

        for( unsigned int col = 0; col < images[0]->columns(); ++col )
        {
            float x = std::abs( srcp[0][col] );
            for( unsigned int i = 1; i < KERNEL_COUNT; ++i )
                x = std::max( x, std::abs( srcp[i][col] ) );

            srcp[0][col] = x;
        }
    }
}
