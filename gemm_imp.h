#ifndef _GEMMM_IMP_H_
#define _GEMMM_IMP_H_



// 基础版本的矩阵乘法
void gemm_v0_MNK(const float *A, const float *B, float *C, int M, int N, int K);

// 置换了矩阵乘法的循环迭代顺序,N放在最内层循环中。B的cache命中提高,但最内层循环中C矩阵中不同位置的访存需求变大
void gemm_v1_MKN(const float *A, const float *B, float *C, int M, int N, int K);

// 对B矩阵进行转置,提升B的cache命中率
void gemm_v2_MNK_transposeB(const float *A, const float *B, float *C, int M, int N, int K);

// 在V1基础上初步用SSE优化,V0的逻辑用SIMD
void gemm_v1_MKN_SSE(const float *A, const float *B, float *C, int M, int N, int K);

// 在V1_0基础上修改。最内层循环做四行和四列。循环展开+减少对C的写回操作
void gemm_v1_MKN_SSE_UNROLL(const float *A, const float *B, float *C, int M, int N, int K);

// v2上用SSE+unroll优化
void gemm_v2_MNK_SSE_UNROLL(const float *A, const float *B, float *C, int M, int N, int K);

// 4大小向量转置B矩阵乘法
void gemm_v2_MNK_SSE_UNROLL_TRANSPOSEV4(const float *A, const float *B, float *C, int M, int N, int K);

// 4大小向量转置矩阵乘法+OMP
void gemm_v2_MNK_SSE_UNROLL_TRANSPOSEV4_OMP(const float *A, const float *B, float *C, int M, int N, int K);

// 4大小向量转置矩阵乘法+OMP+展开2行
void gemm_v2_MNK_SSE_UNROLL2_TRANSPOSEV4_OMP(const float *A, const float *B, float *C, int M, int N, int K);

// 向量转置
// M*N -> 1/4N*4M
void transpose_vec4(const float *A, float *B, int M, int N);

// 转置
// M*N -> N*M
void transpose(const float *A, float *B, int M, int N);

// 矩阵分块乘法
void dgemm_block(const float *A, const float *B, float *C, int M, int N, int K);

// openblas
void gemm_blas(const float *A, const float *B, float *C, int M, int N, int K);

#endif