/** @file
 * @brief Template definition of class IKirschOperatorFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "IImage.hxx"

/*************************************************************************/
/* IKirschOperatorFilter< F >                                            */
/*************************************************************************/
template< typename F >
const float
IKirschOperatorFilter< F >::KERNELS[KERNEL_COUNT][(2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1)] =
{
    { -3.0f, -3.0f,  5.0f,
      -3.0f,  0.0f,  5.0f,
      -3.0f, -3.0f,  5.0f },

    { -3.0f,  5.0f,  5.0f,
      -3.0f,  0.0f,  5.0f,
      -3.0f, -3.0f, -3.0f },

    {  5.0f,  5.0f,  5.0f,
      -3.0f,  0.0f, -3.0f,
      -3.0f, -3.0f, -3.0f },

    {  5.0f,  5.0f, -3.0f,
       5.0f,  0.0f, -3.0f,
      -3.0f, -3.0f, -3.0f },

    {  5.0f, -3.0f, -3.0f,
       5.0f,  0.0f, -3.0f,
       5.0f, -3.0f, -3.0f },

    { -3.0f, -3.0f, -3.0f,
       5.0f,  0.0f, -3.0f,
       5.0f,  5.0f, -3.0f },

    { -3.0f, -3.0f, -3.0f,
      -3.0f,  0.0f, -3.0f,
       5.0f,  5.0f,  5.0f },

    { -3.0f, -3.0f, -3.0f,
      -3.0f,  0.0f,  5.0f,
      -3.0f,  5.0f,  5.0f },
};

template< typename F >
IKirschOperatorFilter< F >::IKirschOperatorFilter()
{
    for( unsigned int i = 0; i < KERNEL_COUNT; ++i )
        mFilters[i].setKernel( (const float*)&KERNELS[i], KERNEL_RADIUS );
}

template< typename F >
void
IKirschOperatorFilter< F >::filter(
    IImage& image
    )
{
    switch( image.format() )
    {
    case Image::FMT_GRAY_FLOAT32:
        break;

    default:
    case Image::FMT_GRAY_UINT8:
    case Image::FMT_RGB_UINT8:
    case Image::FMT_RGB_FLOAT32:
        throw std::runtime_error(
            "IKirschOperatorFilter: Unsupported image format" );
    }

    IImage* dups[KERNEL_COUNT] =
    {
        &image,        image.clone(),
        image.clone(), image.clone(),
        image.clone(), image.clone(),
        image.clone(), image.clone()
    };

    for( unsigned int i = 0; i < KERNEL_COUNT; ++i )
        mFilters[i].filter( *dups[i] );

    computeGradient( dups );

    for( unsigned int i = 1; i < KERNEL_COUNT; ++i )
        delete dups[i];
}
