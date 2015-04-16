/** @file
 * @brief Template definition of class GaussianBlurFilter.
 *
 * @author Jan Bobek
 */

/*************************************************************************/
/* GaussianBlurFilter< RCF, CCF >                                        */
/*************************************************************************/
template< typename RCF, typename CCF >
void
GaussianBlurFilter< RCF, CCF >::setRadius(
    unsigned int radius
    )
{
    const unsigned int origin = radius;
    const unsigned int length = 2 * radius + 1;

    const double sigma = radius / 2.5;
    const double coef = -1.0 / (2.0 * sigma * sigma);

    mKernel = (float*)realloc(
        mKernel, length * sizeof(*mKernel) );

    float sum = mKernel[origin] = 1.0f;
    for( unsigned int i = 1, r2i = 1; i <= radius; r2i += 1 + 2 * i++ )
        sum += 2.0f *
            (mKernel[origin - i] =
             mKernel[origin + i] =
             exp( coef * r2i ));

    for( unsigned int i = 0; i < length; ++i )
        mKernel[i] /= sum;

    mFilter.setRowKernel( mKernel, radius );
    mFilter.setColumnKernel( mKernel, radius );
}
