/** @file
 * @brief Declaration of class ISobelOperatorFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef ISOBEL_OPERATOR_FILTER_HXX__INCL__
#define ISOBEL_OPERATOR_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a Sobel operator filter.
 *
 * @author Jan Bobek
 */
template< typename F >
class ISobelOperatorFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the filter.
     */
    ISobelOperatorFilter();

    /**
     * @brief Applies the Sobel operator to the image.
     *
     * @param[in,out] image
     *   The image to apply the operator to.
     */
    void filter( IImage& image );

protected:
    /**
     * @brief Computes gradient as given
     *   by Sobel operator.
     *
     * @param[in,out] vert
     *   Detected vertical edges.
     * @param[in] horz
     *   Detected horizontal edges.
     */
    virtual void computeGradient( IImage& vert, const IImage& horz ) = 0;

    /// The filter we use for vertical edges.
    F mVertFilter;
    /// The filter we use for horizontal edges.
    F mHorzFilter;

    /// Radius of both Sobel kernels.
    static const unsigned int KERNEL_RADIUS = 1;
    /// The (-1 0 1) component of Sobel kernel.
    static const float KERNEL_1_0_1[2 * KERNEL_RADIUS + 1];
    /// The (1 2 1) component of Sobel kernel.
    static const float KERNEL_1_2_1[2 * KERNEL_RADIUS + 1];
};

#include "ISobelOperatorFilter.txx"

#endif /* !ISOBEL_OPERATOR_FILTER_HXX__INCL__ */
