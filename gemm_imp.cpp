#include"gemm_imp.h"
#include<iostream>
#include <immintrin.h>     //SSE
#include <xmmintrin.h>     //SSE
#include <emmintrin.h>     //SSE2
#include <pmmintrin.h>     //SSE3
#include <tmmintrin.h>     //SSSE3
#include<assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// 基础版本的矩阵乘法
void gemm_v0_MNK(const float *A, const float *B, float *C, int M, int N, int K)
{
	memset(C, 0, M*N*sizeof(float));
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n++)
		{
			for (int k = 0; k < K; k++)
			{
				C[m*N + n] += A[m*K + k] * B[k*N + n];
			}
		}
	}
	return;
}

// 置换了矩阵乘法的循环迭代顺序,N放在最内层循环中。B的cache命中提高,但最内层循环中C矩阵中不同位置的访存需求变大
void gemm_v1_MKN(const float *A, const float *B, float *C, int M, int N, int K)
{
	memset(C, 0, M*N*sizeof(float));
	for (int m = 0; m < M; m++)
	{
		for (int k = 0; k < K; k++)
		{
			float a = A[m*K + k];
			for (int n = 0; n < N; n++)
			{
				C[m*N + n] += a* B[k*N + n];
			}
		}
	}
	return;
}

// 对B矩阵进行转置,提升B的cache命中率
void gemm_v2_MNK_transposeB(const float *A, const float *B, float *C, int M, int N, int K)
{
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n++)
		{
			float sum = 0.0f;
			for (int k = 0; k < K; k++)
			{
				sum += A[m*K + k] * B[n*K + k];
			}
			C[m*N + n] = sum;
		}
	}
	return;
}

// 在V1基础上初步用SSE优化,V0的逻辑用SIMD
void gemm_v1_MKN_SSE(const float *A, const float *B, float *C, int M, int N, int K)
{
	memset(C, 0, M*N*sizeof(float));
	int m, n, k;
	for (m = 0; m < M; m++)
	{
		for (k = 0; k < K; k++)
		{
			__m128 v4_a = _mm_set1_ps(*(A + m*K + k));// amk amk amk amk
			for (n = 0; n < N - 3; n += 4)
			{
				__m128 v4_b = _mm_loadu_ps(B + k*N + n); // bkn bkn+1 bkn+2 bkn+3
				__m128 v4_c = _mm_loadu_ps(C + m*N + n);
				_mm_storeu_ps(C + m*N + n, _mm_add_ps(v4_c, _mm_mul_ps(v4_a, v4_b)));
			}
			for (; n < N; n++)
			{
				C[m*N + n] += A[m*K + k] * B[k*N + n];
			}
		}
	}
	return;
}

// 在V1_0基础上修改。最内层循环做四行和四列。循环展开+减少对C的写回操作
void gemm_v1_MKN_SSE_UNROLL(const float *A, const float *B, float *C, int M, int N, int K)
{
	memset(C, 0, M*N*sizeof(float));
	int m, n, k;
	for (m = 0; m < M; m++)
	{
		for (k = 0; k < K - 3; k += 4)
		{
			__m128 v4_a0 = _mm_set1_ps(*(A + m*K + k));
			__m128 v4_a1 = _mm_set1_ps(*(A + m*K + k + 1));
			__m128 v4_a2 = _mm_set1_ps(*(A + m*K + k + 2));
			__m128 v4_a3 = _mm_set1_ps(*(A + m*K + k + 3));
			for (n = 0; n < N - 3; n += 4)
			{
				__m128 v4_b0 = _mm_loadu_ps(B + k*N + n);
				__m128 v4_b1 = _mm_loadu_ps(B + k*N + n + N);
				__m128 v4_b2 = _mm_loadu_ps(B + k*N + n + 2 * N);
				__m128 v4_b3 = _mm_loadu_ps(B + k*N + n + 3 * N);

				__m128 v4_c = _mm_loadu_ps(C + m*N + n);
				v4_c = _mm_add_ps(v4_c, _mm_mul_ps(v4_a0, v4_b0));
				v4_c = _mm_add_ps(v4_c, _mm_mul_ps(v4_a1, v4_b1));
				v4_c = _mm_add_ps(v4_c, _mm_mul_ps(v4_a2, v4_b2));
				v4_c = _mm_add_ps(v4_c, _mm_mul_ps(v4_a3, v4_b3));
				_mm_storeu_ps(C + m*N + n, v4_c);
			}
			for (; n < N; n++)
			{
				C[m*N + n] += A[m*K + k] * B[k*N + n];
				C[m*N + n] += A[m*K + k + 1] * B[(k + 1)*N + n];
				C[m*N + n] += A[m*K + k + 2] * B[(k + 2)*N + n];
				C[m*N + n] += A[m*K + k + 3] * B[(k + 3)*N + n];
			}
		}
		for (; k < K; k++)
		{
			__m128 v4_a0 = _mm_set1_ps(*(A + m*K + k));

			for (n = 0; n < N - 3; n += 4)
			{
				__m128 v4_b = _mm_loadu_ps(B + k*N + n);
				__m128 v4_c = _mm_loadu_ps(C + m*N + n);
				_mm_storeu_ps(C + m*N + n, _mm_add_ps(v4_c, _mm_mul_ps(v4_a0, v4_b)));
			}

			float a = A[m*K + k];
			for (; n < N; n++)
			{
				C[m*N + n] += a* B[k*N + n];
			}
		}
	}
	return;
}

// v2 transposeB基础上上用SSE+unroll优化
void gemm_v2_MNK_SSE_UNROLL(const float *A, const float *B, float *C, int M, int N, int K)
{
	int k = 0, n = 0;
	__m128 v4_1_ps = _mm_set1_ps(1.0f);
	__m128 v4_sum_tmp_ps, v4_sumv_tmp_ps;
	for (int m = 0; m < M; m++)
	{
		for (n = 0; n < N - 3; n += 4)
		{
			float sum0, sum1, sum2, sum3;
			__m128 v4_sum0 = _mm_setzero_ps();
			__m128 v4_sum1 = _mm_setzero_ps();
			__m128 v4_sum2 = _mm_setzero_ps();
			__m128 v4_sum3 = _mm_setzero_ps();

			sum0 = sum1 = sum2 = sum3 = 0.0f;
			for (k = 0; k < K - 3; k += 4)
			{
				__m128 a = _mm_loadu_ps(A + m*K + k);

				__m128 b0 = _mm_loadu_ps(B + n*K + k);
				__m128 b1 = _mm_loadu_ps(B + n*K + k + K);
				__m128 b2 = _mm_loadu_ps(B + n*K + k + 2 * K);
				__m128 b3 = _mm_loadu_ps(B + n*K + k + 3 * K);

				v4_sum0 = _mm_add_ps(v4_sum0, _mm_mul_ps(a, b0));
				v4_sum1 = _mm_add_ps(v4_sum1, _mm_mul_ps(a, b1));
				v4_sum2 = _mm_add_ps(v4_sum2, _mm_mul_ps(a, b2));
				v4_sum3 = _mm_add_ps(v4_sum3, _mm_mul_ps(a, b3));
			}
			for (; k < K; k++)
			{
				sum0 += A[m*K + k] * B[n*K + k];
				sum1 += A[m*K + k] * B[n*K + k + K];
				sum2 += A[m*K + k] * B[n*K + k + 2 * k];
				sum3 += A[m*K + k] * B[n*K + k + 3 * k];
			}
			v4_sum_tmp_ps = _mm_setr_ps(sum0, sum1, sum2, sum3);

			//v4_sumv_tmp_ps.m128_f32[0] = v4_sum0.m128_f32[0] + v4_sum0.m128_f32[1] + v4_sum0.m128_f32[2] + v4_sum0.m128_f32[3];
			v4_sumv_tmp_ps = _mm_dp_ps(v4_sum0, v4_1_ps, 0xF1);
			v4_sum_tmp_ps = _mm_add_ps(v4_sum_tmp_ps, v4_sumv_tmp_ps);

			v4_sumv_tmp_ps = _mm_dp_ps(v4_sum1, v4_1_ps, 0xF2);
			v4_sum_tmp_ps = _mm_add_ps(v4_sum_tmp_ps, v4_sumv_tmp_ps);

			v4_sumv_tmp_ps = _mm_dp_ps(v4_sum2, v4_1_ps, 0xF4);
			v4_sum_tmp_ps = _mm_add_ps(v4_sum_tmp_ps, v4_sumv_tmp_ps);

			v4_sumv_tmp_ps = _mm_dp_ps(v4_sum3, v4_1_ps, 0xF8);
			v4_sum_tmp_ps = _mm_add_ps(v4_sum_tmp_ps, v4_sumv_tmp_ps);

			_mm_storeu_ps(C + m*N + n, v4_sum_tmp_ps);
		}//end for n=0~N-3
		for (; n < N; n++)
		{
			float sum0;
			__m128 v4_sum0 = _mm_setzero_ps();
			sum0 = 0.0f;
			for (k = 0; k < K - 3; k += 4)
			{
				__m128 a = _mm_loadu_ps(A + m*K + k);
				__m128 b0 = _mm_loadu_ps(B + n*K + k);
				v4_sum0 = _mm_add_ps(v4_sum0, _mm_mul_ps(a, b0));
			}
			for (; k < K; k++)
			{
				sum0 += A[m*K + k] * B[n*K + k];
			}
			C[m*N + n] = sum0 + v4_sum0.m128_f32[0] + v4_sum0.m128_f32[1] + v4_sum0.m128_f32[2] + v4_sum0.m128_f32[3];
		}//end for n=N-3~N
	}// end for m
	return;
}

// 向量转置
// M*N -> 1/4N*4M
void transpose_vec4(const float *A, float *B, int M, int N)
{
	int m, n;
	for (m = 0; m < M; m++)
	{
		for (n = 0; n < N; n += 4)
		{
			__m128 a = _mm_loadu_ps(A + m*N + n);
			_mm_storeu_ps(B + n*M + (m << 2), a);
		}
	}
}

// 4大小向量转置B矩阵乘法
void gemm_v2_MNK_SSE_UNROLL_TRANSPOSEV4(const float *A, const float *B, float *C, int M, int N, int K)
{
	assert(0 == N % 4);
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n += 4)
		{
			__m128 v4_sum = _mm_set1_ps(0.0f);
			const float* pA = A + m*K;
			const float* pB = B + n*K;
			int k;
			for (k = 0; k < K - 3; k += 4)
			{
				__m128 v4_a0 = _mm_load1_ps(pA);
				__m128 v4_a1 = _mm_load1_ps(pA + 1);
				__m128 v4_a2 = _mm_load1_ps(pA + 2);
				__m128 v4_a3 = _mm_load1_ps(pA + 3);


				__m128 v4_b0 = _mm_loadu_ps(pB);
				__m128 v4_b1 = _mm_loadu_ps(pB + 4);
				__m128 v4_b2 = _mm_loadu_ps(pB + 8);
				__m128 v4_b3 = _mm_loadu_ps(pB + 12);

				__m128 v4_c = _mm_mul_ps(v4_a0, v4_b0);
				v4_sum = _mm_add_ps(v4_sum, v4_c);

				v4_c = _mm_mul_ps(v4_a1, v4_b1);
				v4_sum = _mm_add_ps(v4_sum, v4_c);

				v4_c = _mm_mul_ps(v4_a2, v4_b2);
				v4_sum = _mm_add_ps(v4_sum, v4_c);

				v4_c = _mm_mul_ps(v4_a3, v4_b3);
				v4_sum = _mm_add_ps(v4_sum, v4_c);

				pA += 4;
				pB += 16;
			}
			for (; k < K; k++)
			{
				__m128 v4_a0 = _mm_load1_ps(pA);

				__m128 v4_b0 = _mm_loadu_ps(pB);

				__m128 v4_c = _mm_mul_ps(v4_a0, v4_b0);
				v4_sum = _mm_add_ps(v4_sum, v4_c);

				pA += 1;
				pB += 4;
			}
			_mm_storeu_ps(C + m*N + n, v4_sum);
		}
	}
	return;
}


// 4大小向量转置矩阵乘法+OMP
void gemm_v2_MNK_SSE_UNROLL_TRANSPOSEV4_OMP(const float *A, const float *B, float *C, int M, int N, int K)
{
	assert(0 == N % 4);
#ifdef _OPENMP 
	omp_set_num_threads(4);
#pragma omp parallel for 
#endif
	for (int m = 0; m < M; m++)
	{
		for (int n = 0; n < N; n += 4)
		{
			__m128 v4_sum = _mm_set1_ps(0.0f);
			const float* pA = A + m*K;
			const float* pB = B + n*K;
			int k;
			for (k = 0; k < K - 3; k += 4)
			{

				__m128 v4_a0 = _mm_load1_ps(pA);
				__m128 v4_b0 = _mm_loadu_ps(pB);
				__m128 v4_c = _mm_mul_ps(v4_a0, v4_b0);
				v4_sum = _mm_add_ps(v4_sum, v4_c);

				__m128 v4_a1 = _mm_load1_ps(pA + 1);
				__m128 v4_b1 = _mm_loadu_ps(pB + 4);
				v4_c = _mm_mul_ps(v4_a1, v4_b1);
				v4_sum = _mm_add_ps(v4_sum, v4_c);


				__m128 v4_a2 = _mm_load1_ps(pA + 2);
				__m128 v4_b2 = _mm_loadu_ps(pB + 8);

				v4_c = _mm_mul_ps(v4_a2, v4_b2);
				v4_sum = _mm_add_ps(v4_sum, v4_c);
				
				__m128 v4_a3 = _mm_load1_ps(pA + 3);
				__m128 v4_b3 = _mm_loadu_ps(pB + 12);
				v4_c = _mm_mul_ps(v4_a3, v4_b3);
				v4_sum = _mm_add_ps(v4_sum, v4_c);

				pA += 4;
				pB += 16;
			}
			for (; k < K; k++)
			{
				__m128 v4_a0 = _mm_load1_ps(pA);

				__m128 v4_b0 = _mm_loadu_ps(pB);

				__m128 v4_c = _mm_mul_ps(v4_a0, v4_b0);
				v4_sum = _mm_add_ps(v4_sum, v4_c);

				pA += 1;
				pB += 4;
			}
			_mm_storeu_ps(C + m*N + n, v4_sum);
		}
	}
	return;
}

void gemm_v2_MNK_SSE_UNROLL2_TRANSPOSEV4_OMP(const float *A, const float *B, float *C, int M, int N, int K)
{
#define CAL_ROWX(x) \
	v4_c = _mm_mul_ps(v4_a0, v4_b0); \
	v4_sum##x = _mm_add_ps(v4_sum##x, v4_c); \
	v4_c = _mm_mul_ps(v4_a1, v4_b1);	 \
	v4_sum##x = _mm_add_ps(v4_sum##x, v4_c); \
	v4_c = _mm_mul_ps(v4_a2, v4_b2);	 \
	v4_sum##x = _mm_add_ps(v4_sum##x, v4_c); \
	v4_c = _mm_mul_ps(v4_a3, v4_b3);	 \
	v4_sum##x = _mm_add_ps(v4_sum##x, v4_c);

	assert(0 == N % 4);
	int m = 0;
#ifdef _OPENMP 
	omp_set_num_threads(4);
#pragma omp parallel for lastprivate(m)
#endif
	for (m = 0; m < M - 1; m += 2)
	{
		for (int n = 0; n < N; n += 4)
		{
			__m128 v4_sum0 = _mm_set1_ps(0.0f);
			__m128 v4_sum1 = v4_sum0;
			const float* pA0 = A + m*K;
			const float* pA1 = A + m*K + K;
			const float* pB = B + n*K;
			int k;
			for (k = 0; k < K - 3; k += 4)
			{
				__m128 v4_c;
				// row0
				__m128 v4_a0 = _mm_load1_ps(pA0);
				__m128 v4_a1 = _mm_load1_ps(pA0 + 1);
				__m128 v4_a2 = _mm_load1_ps(pA0 + 2);
				__m128 v4_a3 = _mm_load1_ps(pA0 + 3);


				__m128 v4_b0 = _mm_loadu_ps(pB);
				__m128 v4_b1 = _mm_loadu_ps(pB + 4);
				__m128 v4_b2 = _mm_loadu_ps(pB + 8);
				__m128 v4_b3 = _mm_loadu_ps(pB + 12);

				CAL_ROWX(0)

					// row1
					v4_a0 = _mm_load1_ps(pA1);
				v4_a1 = _mm_load1_ps(pA1 + 1);
				v4_a2 = _mm_load1_ps(pA1 + 2);
				v4_a3 = _mm_load1_ps(pA1 + 3);

				CAL_ROWX(1)

					pA0 += 4;
				pA1 += 4;
				pB += 16;
			}
			for (; k < K; k++)
			{
				__m128 v4_a0 = _mm_load1_ps(pA0);
				__m128 v4_a1 = _mm_load1_ps(pA1);

				__m128 v4_b0 = _mm_loadu_ps(pB);

				// row0
				__m128 v4_c = _mm_mul_ps(v4_a0, v4_b0);
				v4_sum0 = _mm_add_ps(v4_sum0, v4_c);

				// row1
				v4_c = _mm_mul_ps(v4_a1, v4_b0);
				v4_sum1 = _mm_add_ps(v4_sum1, v4_c);

				pA0++;
				pA1++;
				pB += 4;
			}
			_mm_storeu_ps(C + m*N + n, v4_sum0);
			_mm_storeu_ps(C + m*N + N + n, v4_sum1);
		}
	}

	// m = M&(-1)
	for (; m < M; m++)
	{
		for (int n = 0; n < N; n += 4)
		{
			__m128 v4_sum0 = _mm_set1_ps(0.0f);
			__m128 v4_c;
			const float* pA0 = A + m*K;
			const float* pB = B + n*K;
			int k;
			for (k = 0; k < K - 3; k += 4)
			{
				// row0
				__m128 v4_a0 = _mm_load1_ps(pA0);
				__m128 v4_a1 = _mm_load1_ps(pA0 + 1);
				__m128 v4_a2 = _mm_load1_ps(pA0 + 2);
				__m128 v4_a3 = _mm_load1_ps(pA0 + 3);


				__m128 v4_b0 = _mm_loadu_ps(pB);
				__m128 v4_b1 = _mm_loadu_ps(pB + 4);
				__m128 v4_b2 = _mm_loadu_ps(pB + 8);
				__m128 v4_b3 = _mm_loadu_ps(pB + 12);

				CAL_ROWX(0)

					pA0 += 4;
				pB += 16;
			}
			for (; k < K; k++)
			{
				__m128 v4_a0 = _mm_load1_ps(pA0);

				__m128 v4_b0 = _mm_loadu_ps(pB);

				// row0
				__m128 v4_c = _mm_mul_ps(v4_a0, v4_b0);
				v4_sum0 = _mm_add_ps(v4_sum0, v4_c);

				pA0++;
				pB += 4;
			}
			_mm_storeu_ps(C + m*N + n, v4_sum0);
		}
	}

	return;
}

void transpose(const float *A, float *B, int M, int N)
{
	for (int n = 0; n < N; n++)
	{
		for (int m = 0; m < M; m++)
		{
			B[n*M + m] = A[N*m + n];
		}
	}
}



inline void do_block(const float *A, const float *B, float *C, int K, int N, int BLOCKSIZE)
{
	
	for (int m = 0; m < BLOCKSIZE; m++)
	{
		for (int n = 0; n < BLOCKSIZE; n++)
		{
			float c = C[m*N + n];
			for (int k = 0; k < BLOCKSIZE; k++)
				c += A[m*K + k] * B[k*N + n];
			C[m*N + n] = c;
		}
	}
}


// 矩阵分块乘法
void dgemm_block(const float *A, const float *B, float *C, int M, int N, int K)
{
	const int BLOCKSIZE = 64;
	memset(C, 0, M*N*sizeof(float));
	for (int m = 0; m < M; m += BLOCKSIZE)
	{
		for (int n = 0; n < N; n += BLOCKSIZE)
		{
			for (int k = 0; k < K; k += BLOCKSIZE)
			{
				do_block(A + m*K + k, B + k*N + n, C + m*N + n, K, N, BLOCKSIZE);
			}
		}
	}
	return;

}


#ifdef USE_OPEN_BLAS

#include"third-party/blas/include/cblas.h"
#pragma comment(lib, "third-party/blas/lib/libopenblas.lib") 

void gemm_blas(const float *A, const float *B, float *C, int M, int N, int K)
{
	const enum CBLAS_ORDER Order = CblasRowMajor;
	const enum CBLAS_TRANSPOSE TransA = CblasNoTrans;
	const enum CBLAS_TRANSPOSE TransB = CblasNoTrans;
	const float alpha = 1;
	const float beta = 0;
	const int lda = K;
	const int ldb = N;
	const int ldc = N;

	cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#else

void gemm_blas(const float *A, const float *B, float *C, int M, int N, int K)
{
	return;
}

#endif
