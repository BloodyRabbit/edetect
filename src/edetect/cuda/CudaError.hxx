/** @file
 * @brief Declarations related to CUDA error handling.
 *
 * @author Jan Bobek
 */

#ifndef CUDA__CUDA_ERROR_HXX__INCL__
#define CUDA__CUDA_ERROR_HXX__INCL__

/**
 * @brief Thrown on CUDA error.
 *
 * @author Jan Bobek
 */
class CudaError
: public std::runtime_error
{
public:
    /**
     * @brief Initializes the exception.
     *
     * @param[in] err
     *   The error code.
     * @param[in] msg
     *   Custom message attached to the exception.
     * @param[in] file
     *   File at which the exception was thrown.
     * @param[in] line
     *   Line at which the exception was thrown.
     */
    CudaError(
        cudaError_t err, const char* msg,
        const char* file, unsigned int line
        );

    /**
     * @brief Obtains the error code.
     *
     * @return
     *   The error code.
     */
    cudaError_t err() const { return mErr; }
    /**
     * @brief Obtains the file at which the exception was thrown.
     *
     * @return
     *   The file at which the exception was thrown.
     */
    const char* file() const { return mFile; }
    /**
     * @brief Obtains the line at which the exception was thrown.
     *
     * @return
     *   The line at which the exception was thrown.
     */
    unsigned int line() const { return mLine; }

protected:
    /**
     * @brief Composes the proper exception message.
     *
     * @param[in] err
     *   The error code.
     * @param[in] msg
     *   Custom message attached to the exception.
     * @param[in] file
     *   File at which the exception was thrown.
     * @param[in] line
     *   Line at which the exception was thrown.
     *
     * @return
     *   The formatted error message.
     */
    static std::string formatMessage(
        cudaError_t err, const char* msg,
        const char* file, unsigned int line
        );

    /// The error code.
    cudaError_t mErr;

    /// File at which the exception was thrown.
    const char* mFile;
    /// Line number at which the exception was thrown.
    unsigned int mLine;
};

#define cudaMsgCheckError( x, msg ) \
    cudaCheckError__( (x), (msg), __FILE__, __LINE__ )
#define cudaCheckError( x ) \
    cudaMsgCheckError( (x), ("`"#x"' failed") )
#define cudaCheckLastError( msg ) \
    cudaMsgCheckError( cudaGetLastError(), (msg) )

inline void
cudaCheckError__(
    cudaError_t x,
    const char* msg,
    const char* file,
    unsigned int line
    )
{
    if( cudaSuccess != x )
        throw CudaError( x, msg, file, line );
}

#endif /* !CUDA__CUDA_ERROR_HXX__INCL__ */
