#include <vector>

#include <torch/torch.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <torch/extension.h>


#include <iostream>
using namespace std;

__global__ void GEMMLowpKernel(const float* in, const int N, float* out,
                               float scale, float shift, long long qmax, const float* noise, bool enforce_true_zero) {
//   printf("Potting Success\n");
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
      out[i] = in[i];
      if (enforce_true_zero)
        out[i] = (out[i] / scale) + shift;
      else
        out[i] = (out[i] + shift) / scale;
      out[i] += noise[i];
      out[i] = fminf(out[i], qmax);
      out[i] = fmaxf(out[i], 0.);
      out[i] = roundf(out[i]);
      if (enforce_true_zero)
        out[i] = (out[i] - shift) * scale;
      else
        out[i] = out[i] * scale - shift;
  }
}


#define block_count 32
#define thread_per_block 1024
// Wrapper for ATen
at::Tensor float2gemmlowp(at::Tensor in, float range, float offset, int num_bits, bool int_exp, bool enforce_true_zero, at::Tensor noise) {
    if (range <= 0)
        return in;

    int N = in.numel();
    auto out = at::zeros_like(in);
    long long qmax = (0x1l << num_bits) - 1;
    float scale = range / qmax;
    if (int_exp)
        scale = powf(2, int(ceilf(log2f(scale))));
    float zero_point = roundf(-offset / scale);
    float shift = enforce_true_zero ? zero_point : -offset;
    GEMMLowpKernel<<<block_count, thread_per_block>>>(in.data<float>(), N, out.data<float>(), scale, shift, qmax, noise.data<float>(), enforce_true_zero);

    return out;
}

//cublasLtHandle_t ltHandle
void cublasLtGemm(int TA, int TB,
                    int m, int n, int k,
                    const float *ALPHA, 
                    at::Tensor A_gpu, int lda,
                    at::Tensor B_gpu, int ldb,
                    const float *BETA,
                    at::Tensor C_gpu, int ldc,
                    void *workspace,
                    size_t workspaceSize)
{
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t opDesc = NULL;
    cublasLtMatrixLayout_t Adesc=NULL, Bdesc=NULL, Cdesc=NULL;
    //cublasLtMatmulPreference_t preference = NULL;

    //int returnedResults = 0;
    //cublasLtMatmulHeuristicResult_t heuristicResult = {};

    cublasOperation_t transa = (TA ? CUBLAS_OP_T : CUBLAS_OP_N);
    cublasOperation_t transb = (TB ? CUBLAS_OP_T : CUBLAS_OP_N);
    status = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    status = cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    status = cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    //status = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, k, k, lda);
    //status = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, m, k, ldb);
    status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc);

    //status = cublasLtMatmulPreferenceCreate(&preference);
    //status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));

    cublasStatus_t stat = cublasLtMatmul(
                                            ltHandle,
                                            opDesc,
                                            ALPHA,
                                            A_gpu.data_ptr<int8_t>(), Adesc,  // Input
                                            B_gpu.data_ptr<int8_t>(), Bdesc,  // Input
                                            BETA,
                                            C_gpu.data_ptr(), Cdesc, // Bias
                                            C_gpu.data_ptr<float>(), Cdesc,// Output
                                            NULL,
                                            workspace,
                                            workspaceSize,
                                            0);

}

void cublasGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                torch::Tensor A_gpu, int lda,
                torch::Tensor B_gpu, int ldb,
                float BETA,
                torch::Tensor C_gpu, int ldc)
{
    //cout << A_gpu.type() << endl;
    //cout << B_gpu.type() << endl;
    //cout << C_gpu.type() << endl;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    /*
    cudaError_t status = static_cast<cudaError_t>(
                                                    cublasGemmEx(
                                                    handle,
                                                    (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                    (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                    M, N, K,
                                                    &ALPHA,
                                                    A_gpu.data_ptr<int8_t>(), CUDA_R_8I, lda,
                                                    B_gpu.data_ptr<int8_t>(), CUDA_R_8I, ldb,

                                                    &BETA,
                                                    C_gpu.data_ptr<int32_t>(), CUDA_R_32I, ldc,
                                                    CUBLAS_COMPUTE_32I,
                                                    A_gpu.data_ptr<float>(), CUDA_R_32F, lda,
                                                    B_gpu.data_ptr<float>(), CUDA_R_32F, ldb,
                                                    &BETA,
                                                    C_gpu.data_ptr<int32_t>(), CUDA_R_32I, ldc,
                                                    CUBLAS_COMPUTE_32I,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP)
                                                );
//     cudaError_t status = static_cast<cudaError_t>(
//                                                     cublasGemmEx(
//                                                     handle,
//                                                     (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
//                                                     (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
//                                                     M, N, K,
//                                                     &ALPHA,
//                                                     A_gpu.data_ptr<float>(), CUDA_R_32F, lda,
//                                                     B_gpu.data_ptr<float>(), CUDA_R_32F, ldb,
//                                                     &BETA,
//                                                     C_gpu.data_ptr<float>(), CUDA_R_32F, ldc,
//                                                     CUDA_R_32F,
//                                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP)
//                                                 );
//     cublasStatus_t status = cublasGemmEx(
//                                                 handle,
//                                                 (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
//                                                 (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
//                                                 M, N, K,
//                                                 &ALPHA,
//                                                 A_gpu.data_ptr<float>(), CUDA_R_32F, lda,
//                                                 B_gpu.data_ptr<float>(), CUDA_R_32F, ldb,
//                                                 &BETA,
//                                                 C_gpu.data_ptr<float>(), CUDA_R_32F, ldc,
//                                                 CUDA_R_32F,
//                                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    */
    // Cublas Gemm
    cublasStatus_t status = cublasGemmEx(
                                                handle,
                                                (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                M, N, K,
                                                static_cast<const void*>(&ALPHA),
                                                static_cast<const void*>(A_gpu.data_ptr()), CUDA_R_16F, lda,
                                                static_cast<const void*>(B_gpu.data_ptr()), CUDA_R_16F, ldb,
                                                static_cast<const void*>(&BETA),
                                                static_cast<void*>(C_gpu.data_ptr()), CUDA_R_16F, ldc,
                                                CUBLAS_COMPUTE_16F,
                                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // CUBLAS_GEMM_DEFAULT_TENSOR_OP
    //if (status == CUBLAS_STATUS_EXECUTION_FAILED) printf("start 1 Found");
    //if (status == CUBLAS_STATUS_INTERNAL_ERROR) printf("Found");
    //printf("%d\n", status);
//     printf("%s\n", cudaGetErrorName(status));
}
