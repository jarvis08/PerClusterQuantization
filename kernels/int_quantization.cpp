#include <torch/torch.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDABlas.h>
#include <torch/extension.h>


// CUDA declarations
at::Tensor float2gemmlowp(at::Tensor in, float range, float offset, int num_bits, bool int_exp,
                          bool enforce_true_zero, at::Tensor noise);

void cublasGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                at::Tensor A_gpu, int lda,
                at::Tensor B_gpu, int ldb,
                float BETA,
                at::Tensor C_gpu, int ldc);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float2gemmlowp", &float2gemmlowp, "Convert float 32 to gemmlowp");
    m.def("cublasGemm", &cublasGemm, "gemm using cublas");
}
