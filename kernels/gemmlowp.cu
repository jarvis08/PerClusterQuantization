#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <torch/torch.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDABlas.h>
#include <torch/extension.h>


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

void cublasGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                at::Tensor A_gpu, int lda,
                at::Tensor B_gpu, int ldb,
                float BETA,
                at::Tensor C_gpu, int ldc)
{


    cublasHandle_t handle;
    cublasCreate(&handle);
    printf("some : %d\n", CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaError_t status = static_cast<cudaError_t>(
                                                    cublasGemmEx(
                                                    handle,
                                                    (TA ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                    (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                    M, N, K,
                                                    &ALPHA,
                                                    A_gpu.data_ptr<uint8_t>(), CUDA_R_8U, lda,
                                                    B_gpu.data_ptr<uint8_t>(), CUDA_R_8U, ldb,
                                                    &BETA,
                                                    C_gpu.data_ptr<int>(), CUDA_R_32I, ldc,
                                                    CUDA_R_32F,
                                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP)
                                                );

    printf("%d\n", status);
    printf("%s\n", cudaGetErrorName(status));
}
