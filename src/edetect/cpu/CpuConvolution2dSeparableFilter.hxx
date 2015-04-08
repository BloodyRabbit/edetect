/** @file
 * @brief Declaration of class CpuConvolution2dSeparableFilter.
 *
 * @author Jan Bobek
 * @since 11th April 2015
 */

#ifndef CPU__CPU_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__
#define CPU__CPU_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__

#include "IConvolution2dSeparableFilter.hxx"

/**
 * @brief CPU-backed 2D separable convolution filter.
 *
 * @author Jan Bobek
 */
class CpuConvolution2dSeparableFilter
: public IConvolution2dSeparableFilter
{
protected:
    /// @copydoc IConvolution2dSeparableFilter::convolveRows(IImage&, const IImage&)
    void convolveRows( IImage& dest, const IImage& src );
    /// @copydoc IConvolution2dSeparableFilter::convolveColumns(IImage&, const IImage&)
    void convolveColumns( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_CONVOLUTION_2D_SEPARABLE_FILTER_HXX__INCL__ */
