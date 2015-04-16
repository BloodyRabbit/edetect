/** @file
 * @brief Declaration of class IKirschOperatorFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef IKIRSCH_OPERATOR_FILTER_HXX__INCL__
#define IKIRSCH_OPERATOR_FILTER_HXX__INCL__

#include "IImageFilter.hxx"

/**
 * @brief Interface of a Kirsch operator filter.
 *
 * @author Jan Bobek
 */
template< typename CF >
class IKirschOperatorFilter
: public IImageFilter
{
public:
    /**
     * @brief Initializes the Kirsch operator.
     */
    IKirschOperatorFilter();

    /**
     * @brief Applies the Kirsch operator to an image.
     *
     * @param[in,out] image
     *   The image to apply the operator to.
     */
    void filter( IImage& image );

protected:
    /// Number of kernels used by Kirsch operator.
    static const unsigned int KERNEL_COUNT = 8;
    /// Radius of kernels used by Kirsch operator.
    static const unsigned int KERNEL_RADIUS = 1;
    /// The kernels used by Kirsch operator.
    static const float KERNELS[KERNEL_COUNT][(2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1)];

    /**
     * @brief Computes the gradient as given
     *   by the Kirsch operator.
     *
     * @param[in,out] images
     *   The images to compute the gradient from.
     */
    virtual void computeGradient(
        IImage* images[KERNEL_COUNT]
        ) = 0;

    /// The filters we use for each kernel.
    CF mFilters[KERNEL_COUNT];
};

#include "IKirschOperatorFilter.txx"

#endif /* !IKIRSCH_OPERATOR_FILTER_HXX__INCL__ */
