/** @file
 * @brief Declaration of CpuKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CPU__CPU_KIRSCH_OPERATOR_FILTER_HXX__INCL__
#define CPU__CPU_KIRSCH_OPERATOR_FILTER_HXX__INCL__

#include "IKirschOperatorFilter.hxx"
#include "cpu/CpuConvolution2dFilter.hxx"

/**
 * @brief Applies Kirsch operator to the image.
 *
 * @author Jan Bobek
 */
class CpuKirschOperatorFilter
: public IKirschOperatorFilter<
    CpuConvolution2dFilter >
{
protected:
    /// @copydoc IKirschOperatorFilter< T >::computeGradient( IImage*[KERNEL_COUNT] )
    void computeGradient( IImage* images[KERNEL_COUNT] );
};

#endif /* !CPU__CPU_KIRSCH_OPERATOR_FILTER_HXX__INCL__ */
