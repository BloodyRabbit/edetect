/** @file
 * @brief Template definition of class GaussianBlurFilter.
 *
 * @author Jan Bobek
 */

/*************************************************************************/
/* GaussianBlurFilter< CF >                                              */
/*************************************************************************/
template< typename CF >
float*
GaussianBlurFilter< CF >::generateKernel(
    unsigned int radius
    )
{
    const unsigned int origin = radius;
    const unsigned int length = 2 * radius + 1;

    const double sigma = radius / 2.5;
    const double coef = -1.0 / (2.0 * sigma * sigma);

    float* const kernel = new float[length];

    float sum = kernel[origin] = 1.0f;
    for( unsigned int i = 1, r2i = 1; i <= radius; r2i += 1 + 2 * i++ )
        sum += 2.0f *
            (kernel[origin - i] =
             kernel[origin + i] =
             exp( coef * r2i ));

    for( unsigned int i = 0; i < length; ++i )
        kernel[i] /= sum;

    return kernel;
}
