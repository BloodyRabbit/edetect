/** @file
 * @brief Declaration of CpuConvolution2dFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CPU__CPU_CONVOLUTION_2D_FILTER_HXX__INCL__
#define CPU__CPU_CONVOLUTION_2D_FILTER_HXX__INCL__

#include "IConvolution2dFilter.hxx"

/**
 * @brief CPU-backed 2D discrete convolution filter.
 *
 * @author Jan Bobek
 */
class CpuConvolution2dFilter
: public IConvolution2dFilter
{
protected:
    /// @copydoc IConvolution2dFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_CONVOLUTION_2D_FILTER_HXX__INCL__ */
