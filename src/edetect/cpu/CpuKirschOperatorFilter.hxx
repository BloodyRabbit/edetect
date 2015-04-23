/** @file
 * @brief Declaration of CpuKirschOperatorFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CPU__CPU_KIRSCH_OPERATOR_FILTER_HXX__INCL__
#define CPU__CPU_KIRSCH_OPERATOR_FILTER_HXX__INCL__

#include "filters/IKirschOperatorFilter.hxx"

/**
 * @brief Applies Kirsch operator to the image.
 *
 * @author Jan Bobek
 */
class CpuKirschOperatorFilter
: public IKirschOperatorFilter
{
protected:
    /// @copydoc IKirschOperatorFilter::applyKirschOperator(IImage&, const IImage&)
    void applyKirschOperator( IImage& dest, const IImage& src );
};

#endif /* !CPU__CPU_KIRSCH_OPERATOR_FILTER_HXX__INCL__ */
