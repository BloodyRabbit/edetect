/** @file
 * @brief Template definition of class DerivativeOfGaussianFilter.
 *
 * @author Jan Bobek
 * @since 18th April 2015
 */

/*************************************************************************/
/* DerivativeOfGaussianFilter< CF >                                      */
/*************************************************************************/
template< typename CF >
float*
DerivativeOfGaussianFilter< CF >::generateKernel(
    unsigned int radius,
    unsigned int& length
    )
{
    const unsigned int origin = radius;
    length = 2 * radius + 1;

    const double sigma = radius / 2.5;
    const double coef = -1.0 / (2.0 * sigma * sigma);

    float* const kernel = (float*)malloc( length * sizeof(*kernel) );

    float sum = kernel[origin] = 0.0f;
    for( unsigned int i = 1, r2i = 1; i <= radius; r2i += 1 + 2 * i++ )
    {
        kernel[origin + i] = exp( coef * r2i );
        kernel[origin - i] = -kernel[origin + i];

        sum += kernel[origin + i];
    }

    for( unsigned int i = 0; i < length; ++i )
        kernel[i] /= sum;

    return kernel;
}
