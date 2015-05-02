/** @file
 * @brief Declaration of class CudaRingBuffer.
 *
 * @author Jan Bobek
 * @since 2nd May 2015
 */

#ifndef CUDA__CUDA_RING_BUFFER_CUH__INCL__
#define CUDA__CUDA_RING_BUFFER_CUH__INCL__

/**
 * @brief A lock-free CUDA implementation
 *   of a ring buffer.
 *
 * @author Jan Bobek
 */
template< typename T >
class CudaRingBuffer
{
public:
    /**
     * @brief Creates a new buffer.
     *
     * @param[in] capacity
     *   Desired capacity of the ring buffer.
     *
     * @return
     *   The created buffer.
     */
    static CudaRingBuffer* create( unsigned int capacity );
    /**
     * @brief Destroys a created buffer.
     *
     * @param[in] buffer
     *   The buffer to destroy.
     */
    static void destroy( CudaRingBuffer* buffer );

    /**
     * @brief Pops a value from the front
     *   of the buffer.
     *
     * @param[out] value
     *   Where to store the value.
     *
     * @retval true
     *   The value has been popped.
     * @retval false
     *   The buffer is empty.
     */
    __device__ bool pop_front( T& value );
    /**
     * @brief Pushes a value to the back
     *   of the buffer.
     *
     * @param[in] value
     *   The value to push.
     */
    __device__ void push_back( const T& value );

protected:
    /**
     * @brief Initializes the buffer.
     *
     * @param[in] capacity
     *   Desired capacity of the buffer.
     */
    CudaRingBuffer( unsigned int capacity );
    /**
     * @brief A protected destructor.
     */
    ~CudaRingBuffer();

    /// Size of the ring buffer.
    unsigned int mSize;
    /// Capacity of the ring buffer.
    unsigned int mCapacity;

    /// Index of the front of the buffer.
    unsigned int mFrontIdx;
    /// Index of the back of the buffer.
    unsigned int mBackIdx;

    /// Contents of the buffer.
    T mData[];
};

#include "cuda/CudaRingBuffer.cut"

#endif /* !CUDA__CUDA_RING_BUFFER_CUH__INCL__ */
