/** @file
 * @brief Declaration of CudaFilterPipeline class.
 *
 * @author Jan Bobek
 */

#ifndef CUDA_FILTER_PIPELINE_HXX__INCL__
#define CUDA_FILTER_PIPELINE_HXX__INCL__

#include "CudaFilter.hxx"

/**
 * @brief A series of filters.
 *
 * @author Jan Bobek
 */
class CudaFilterPipeline
: public CudaFilter
{
public:
    /**
     * @brief Releases the filters.
     */
    ~CudaFilterPipeline();

    /**
     * @brief Adds a filter to the pipeline.
     *
     * @param[in] filter
     *   The filter to add.
     */
    void add( CudaFilter* filter );
    /**
     * @brief Clears filters in the pipeline.
     */
    void clear();

    /**
     * @brief Applies the filters to an image.
     *
     * @param[in,out] image
     *   The image to filter.
     */
    void process( CudaImage& image );

protected:
    /// The list of active filters.
    std::list< CudaFilter* > mFilters;
};

#endif /* !CUDA_FILTER_PIPELINE_HXX__INCL__ */
