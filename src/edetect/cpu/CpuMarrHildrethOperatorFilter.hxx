/** @file
 * @brief Declaration of CpuMarrHildrethOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CPU__CPU_MARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__
#define CPU__CPU_MARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__

#include "IMarrHildrethOperatorFilter.hxx"
#include "cpu/CpuConvolutionFilter.hxx"
#include "cpu/CpuZeroCrossFilter.hxx"

/**
 * @brief Applies Marr-Hildreth operator to the image.
 *
 * @author Jan Bobek
 */
class CpuMarrHildrethOperatorFilter
: public IMarrHildrethOperatorFilter<
    CpuConvolutionFilter,
    CpuZeroCrossFilter >
{
protected:
    /// @copydoc IMarrHildrethOperatorFilter< F >::mergeEdges(IImage&, const IImage&)
    void mergeEdges( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_MARR_HILDRETH_OPERATOR_FILTER_HXX__INCL__ */
