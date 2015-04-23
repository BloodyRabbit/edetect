/** @file
 * @brief Declaration of CudaIntFloatFilter class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__
#define CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__

#include "filters/IIntFloatFilter.hxx"

/**
 * @brief Converts integer-pixel images to
 *   float-pixel and vice versa.
 *
 * @author Jan Bobek
 */
class CudaIntFloatFilter
: public IIntFloatFilter
{
protected:
    /// @copydoc IIntFloatFilter::convertInt2Float(IImage&, const IImage&)
    void convertInt2Float( IImage& dest, const IImage& src );
    /// @copydoc IIntFloatFilter::convertFloat2Int(IImage&, const IImage&)
    void convertFloat2Int( IImage& dest, const IImage& src );
};

#endif /* !CUDA__CUDA_INT_FLOAT_FILTER_HXX__INCL__ */
