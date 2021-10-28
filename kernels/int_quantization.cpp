#include <torch/torch.h>

#include <cuda.h>
#include <cublasLt.h>
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

void cublasLtGemm(int TA, int TB,      
                    int m, int n, int k, 
                    const float *ALPHA, 
                    at::Tensor A_gpu, int lda, 
                    at::Tensor B_gpu, int ldb, 
                    const float *BETA,
                    at::Tensor C_gpu, int ldc,                                   
                    void *workspace,   
                    size_t workspaceSize);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float2gemmlowp", &float2gemmlowp, "Convert float 32 to gemmlowp");
    m.def("cublasGemm", &cublasGemm, "gemm using cublas");
    m.def("cublasLtGemm", &cublasLtGemm, "Ltgemm using cublas");
}
