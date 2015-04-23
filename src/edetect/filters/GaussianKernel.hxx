/** @file
 * @brief Declaration of class GaussianKernel and relatives.
 *
 * @author Jan Bobek
 * @since 23th April 2015
 */

#ifndef FILTERS__GAUSSIAN_KERNEL_HXX__INCL__
#define FILTERS__GAUSSIAN_KERNEL_HXX__INCL__

/**
 * @brief Generates a common 1D Gaussian kernel.
 *
 * @author Jan Bobek
 */
class GaussianKernel
{
public:
    /**
     * @brief Generates the 1D Gaussian kernel.
     *
     * @param[in] radius
     *   Radius of the kernel to generate.
     * @param[out] length
     *   Length of the generated kernel.
     *
     * @return
     *   The generated kernel.
     */
    float* operator()( unsigned int radius, unsigned int& length );
};

/**
 * @brief Generates a 1D Derivative-of-Gaussian kernel.
 *
 * @author Jan Bobek
 */
class DerivativeOfGaussianKernel
{
public:
    /**
     * @brief Generates the 1D Derivativ-of-Gaussian kernel.
     *
     * @param[in] radius
     *   Radius of the kernel to generate.
     * @param[out] length
     *   Length of the generated kernel.
     *
     * @return
     *   The generated kernel.
     */
    float* operator()( unsigned int radius, unsigned int& length );
};

/**
 * @brief Generates a 2D Laplacian-of-Gaussian kernel.
 *
 * @author Jan Bobek
 */
class LaplacianOfGaussianKernel
{
public:
    /**
     * @brief Generates the 2D Laplacian-of-Gaussian kernel.
     *
     * @param[in] radius
     *   Radius of the kernel to generate.
     * @param[out] length
     *   Length of the generated kernel.
     *
     * @return
     *   The generated kernel.
     */
    float* operator()( unsigned int radius, unsigned int& length );
};

#endif /* !FILTERS__GAUSSIAN_KERNEL_HXX__INCL__ */
