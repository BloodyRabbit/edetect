/** @file
 * @brief Definitions related to CUDA error handling.
 *
 * @author Jan Bobek
 */

#include "common.hxx"
#include "cuda/CudaError.hxx"

/*************************************************************************/
/* CudaError                                                             */
/*************************************************************************/
std::string
CudaError::formatMessage(
    cudaError_t err,
    const char* msg,
    const char* file,
    unsigned int line
    )
{
    std::ostringstream oss;
    oss << file << ':' << line << ": " << msg
        << ": " << cudaGetErrorString( err );
    return oss.str();
}

CudaError::CudaError(
    cudaError_t err,
    const char* msg,
    const char* file,
    unsigned int line
    )
: std::runtime_error(
    formatMessage( err, msg, file, line ) ),
  mErr( err ),
  mFile( file ),
  mLine( line )
{
}
