/** @file
 * @brief Template definition of class LaplacianOfGaussianFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

/*************************************************************************/
/* LaplacianOfGaussianFilter< CF >                                       */
/*************************************************************************/
template< typename CF >
float*
LaplacianOfGaussianFilter< CF >::generateKernel(
    unsigned int radius,
    unsigned int& length
    )
{
    const unsigned int stride = 2 * radius + 1;
    const unsigned int origin = radius * (stride + 1);
    length = stride * stride;

    const double sigma = radius / 2.5;
    const double coef1 = 2.0 * sigma * sigma;
    const double coef2 = -1.0 / coef1;

    float* const kernel = (float*)malloc( length * sizeof(*kernel) );

    kernel[origin] = -coef1;
    for( unsigned int i = 1, r2i = 1; i <= radius; r2i += 1 + 2 * i++ )
    {
        (kernel[origin - i] =
         kernel[origin - i * stride] =
         kernel[origin + i] =
         kernel[origin + i * stride] =
         (r2i - coef1) * exp( coef2 * r2i ));

        (kernel[origin - i * (stride - 1)] =
         kernel[origin - i * (stride + 1)] =
         kernel[origin + i * (stride - 1)] =
         kernel[origin + i * (stride + 1)] =
         (2 * r2i - coef1) * exp( coef2 * (2 * r2i) ));

        for( unsigned int j = i + 1, r2ij = r2i + j * j; j <= radius; r2ij += 1 + 2 * j++ )
            (kernel[origin - i * stride - j] =
             kernel[origin - i * stride + j] =
             kernel[origin + i * stride - j] =
             kernel[origin + i * stride + j] =

             kernel[origin - j * stride - i] =
             kernel[origin - j * stride + i] =
             kernel[origin + j * stride - i] =
             kernel[origin + j * stride + i] =

             (r2ij - coef1) * exp( coef2 * r2ij ));
    }

    return kernel;
}
