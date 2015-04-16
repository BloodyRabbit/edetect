/** @file
 * @brief Template definition of class ISobelOperatorFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#include "IImage.hxx"

/*************************************************************************/
/* ISobelOperatorFilter< RCF, CCF >                                      */
/*************************************************************************/
template< typename RCF, typename CCF >
const float
ISobelOperatorFilter< RCF, CCF >::KERNEL_1_0_1[] =
{
    -1.0f, 0.0f, 1.0f
};

template< typename RCF, typename CCF >
const float
ISobelOperatorFilter< RCF, CCF >::KERNEL_1_2_1[] =
{
    1.0f, 2.0f, 1.0f
};

template< typename RCF, typename CCF >
ISobelOperatorFilter< RCF, CCF >::ISobelOperatorFilter()
{
    mVertFilter.setRowKernel(
        KERNEL_1_0_1, KERNEL_RADIUS );
    mVertFilter.setColumnKernel(
        KERNEL_1_2_1, KERNEL_RADIUS );

    mHorzFilter.setRowKernel(
        KERNEL_1_2_1, KERNEL_RADIUS );
    mHorzFilter.setColumnKernel(
        KERNEL_1_0_1, KERNEL_RADIUS );
}

template< typename RCF, typename CCF >
void
ISobelOperatorFilter< RCF, CCF >::filter(
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
            "ISobelOperatorFilter: Unsupported image format" );
    }

    IImage* dup = image.clone();
    mVertFilter.filter( image );
    mHorzFilter.filter( *dup );

    computeGradient( image, *dup );
    delete dup;
}
