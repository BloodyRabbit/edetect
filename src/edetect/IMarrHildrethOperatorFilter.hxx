/** @file
 * @brief Declaration of class IMarrHildrethOperatorFilter.
 *
 * @author Jan Bobek
 * @since 12th April 2015
 */

#ifndef IMARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__
#define IMARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__

#include "LaplacianOfGaussianFilter.hxx"

/**
 * @brief Interface of a Marr-Hildreth
 *   operator filter.
 *
 * @author Jan Bobek
 */
template< typename F1, typename F2 >
class IMarrHildrethOperatorFilter
: public IImageFilter
{
public:
    /**
     * @brief Applies the Marr-Hildreth operator
     *   to the image.
     *
     * @param[in,out] image
     *   The image to apply the operator to.
     */
    void filter( IImage& image );
    /// @copydoc IImageFilter::setParam(const char*, const void*)
    void setParam( const char* name, const void* value );

    /**
     * @brief Sets radius of the first filter.
     *
     * @param[in] radius
     *   The desired kernel radius.
     */
    void setRadius1( unsigned int radius );
    /**
     * @brief Sets radius of the second filter.
     *
     * @param[in] radius
     *   The desired kernel radius.
     */
    void setRadius2( unsigned int radius );

protected:
    /**
     * @brief Merges edge pixels into a single image.
     *
     * @param[in,out] dest
     *   Where to place the result.
     * @param[in] src
     *   Source image data.
     */
    virtual void mergeEdges( IImage& dest, const IImage& src ) = 0;

    /// The LoG filters.
    LaplacianOfGaussianFilter< F1 > mLogFilt1, mLogFilt2;
    /// The zero-crossing filter.
    F2 mZeroCrossFilt;
};

#include "IMarrHildrethOperatorFilter.txx"

#endif /* !IMARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__ */
