#ifndef _GEMMM_IMP_H_
#define _GEMMM_IMP_H_



// �����汾�ľ���˷�
void gemm_v0_MNK(const float *A, const float *B, float *C, int M, int N, int K);

// �û��˾���˷���ѭ������˳��,N�������ڲ�ѭ���С�B��cache�������,�����ڲ�ѭ����C�����в�ͬλ�õķô�������
void gemm_v1_MKN(const float *A, const float *B, float *C, int M, int N, int K);

// ��B�������ת��,����B��cache������
void gemm_v2_MNK_transposeB(const float *A, const float *B, float *C, int M, int N, int K);

// ��V1�����ϳ�����SSE�Ż�,V0���߼���SIMD
void gemm_v1_MKN_SSE(const float *A, const float *B, float *C, int M, int N, int K);

// ��V1_0�������޸ġ����ڲ�ѭ�������к����С�ѭ��չ��+���ٶ�C��д�ز���
void gemm_v1_MKN_SSE_UNROLL(const float *A, const float *B, float *C, int M, int N, int K);

// v2����SSE+unroll�Ż�
void gemm_v2_MNK_SSE_UNROLL(const float *A, const float *B, float *C, int M, int N, int K);

// 4��С����ת��B����˷�
void gemm_v2_MNK_SSE_UNROLL_TRANSPOSEV4(const float *A, const float *B, float *C, int M, int N, int K);

// 4��С����ת�þ���˷�+OMP
void gemm_v2_MNK_SSE_UNROLL_TRANSPOSEV4_OMP(const float *A, const float *B, float *C, int M, int N, int K);

// 4��С����ת�þ���˷�+OMP+չ��2��
void gemm_v2_MNK_SSE_UNROLL2_TRANSPOSEV4_OMP(const float *A, const float *B, float *C, int M, int N, int K);

// ����ת��
// M*N -> 1/4N*4M
void transpose_vec4(const float *A, float *B, int M, int N);

// ת��
// M*N -> N*M
void transpose(const float *A, float *B, int M, int N);

// ����ֿ�˷�
void dgemm_block(const float *A, const float *B, float *C, int M, int N, int K);

// openblas
void gemm_blas(const float *A, const float *B, float *C, int M, int N, int K);

#endif