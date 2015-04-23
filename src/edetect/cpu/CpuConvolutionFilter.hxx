/** @file
 * @brief Declaration of CpuConvolutionFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CPU__CPU_CONVOLUTION_FILTER_HXX__INCL__
#define CPU__CPU_CONVOLUTION_FILTER_HXX__INCL__

#include "filters/IConvolutionFilter.hxx"

/**
 * @brief CPU-backed 2D discrete convolution filter.
 *
 * @author Jan Bobek
 */
class CpuConvolutionFilter
: public IConvolutionFilter
{
protected:
    /// @copydoc IConvolutionFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

/**
 * @brief CPU-backed 1D discrete row convolution filter.
 *
 * @author Jan Bobek
 */
class CpuRowConvolutionFilter
: public IConvolutionFilter
{
protected:
    /// @copydoc IConvolutionFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

/**
 * @brief CPU-backed 1D discrete column convolution filter.
 *
 * @author Jan Bobek
 */
class CpuColumnConvolutionFilter
: public IConvolutionFilter
{
protected:
    /// @copydoc IConvolutionFilter::convolve(IImage&, const IImage&)
    void convolve( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_CONVOLUTION_FILTER_HXX__INCL__ */
