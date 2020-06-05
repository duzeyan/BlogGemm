#include<iostream>
#include"timeProfile.h"

#include"global_def.h"
#include"gemm_imp.h"



void setMatVal(float *ptr,int len,float val)
{
	for (int i = 0; i < len; i++)
		ptr[i] = val;
}

void setMatRandVal(float *ptr, int len)
{
	for (int i = 0; i < len; i++)
		ptr[i] = (rand() % 100 - 50)*0.02f;
	
}

int compare(const float *A, const float *B, int len)
{
	if (fabs((*A++) - (*B++)) > 10e-4)
	{
		return 1;
	}
	return 0;
}

const GemmImp gemmProxy[] =
{
	_DECLARE_GEMM_IMP(gemm_v0_MNK, 0),// 0
	_DECLARE_GEMM_IMP(gemm_v1_MKN, 0),
	_DECLARE_GEMM_IMP(dgemm_block, 0),// 2
	_DECLARE_GEMM_IMP(gemm_v2_MNK_transposeB, 1),
	_DECLARE_GEMM_IMP(gemm_v1_MKN_SSE, 0),// 4
	_DECLARE_GEMM_IMP(gemm_v1_MKN_SSE_UNROLL, 0),
	_DECLARE_GEMM_IMP(gemm_v2_MNK_SSE_UNROLL, 1),// 6
	_DECLARE_GEMM_IMP(gemm_v2_MNK_SSE_UNROLL_TRANSPOSEV4, 2),
	_DECLARE_GEMM_IMP(gemm_v2_MNK_SSE_UNROLL_TRANSPOSEV4_OMP, 2),//8
	_DECLARE_GEMM_IMP(gemm_v2_MNK_SSE_UNROLL2_TRANSPOSEV4_OMP, 2),
	_DECLARE_GEMM_IMP(gemm_blas, 0),//10
};


#define RAND_INPUT
void testAll()
{
	int M, N, K;

	M = 512;
	N = 512;
	K = 512;
	int iterMax = 1;

	float *A = new float[M*K];
	float *B = new float[K*N];
	float *B1 = new float[K*N];
	float *C = new float[M*N];
	float *C1 = new float[M*N];

#ifdef RAND_INPUT
	setMatRandVal(A, M*K);
	setMatRandVal(B, K*N);
#else
	setMatVal(A, M*K, 1.0f);
	setMatVal(B, K*N, 2.0f);
#endif
	
	// run benchmark anyway
	gemm_v0_MNK(A, B, C, M, N, K);

	// run vary versions
	int select[] = {9};// index in gemmProxy
	for (int i = 0; i < sizeof(select) / 4; i++)
	{
		{
			UnitTimeProfile timer(gemmProxy[select[i]].pStrDescription, iterMax);
			for (int iter = 0; iter < iterMax; iter++)
			{

				if (1 == gemmProxy[select[i]].transType)
				{
					transpose(B, B1, K, N);
					gemmProxy[select[i]].pFunImp(A, B1, C1, M, N, K);
				}
				else if (2 == gemmProxy[select[i]].transType)
				{
					transpose_vec4(B, B1, K, N);
					gemmProxy[select[i]].pFunImp(A, B1, C1, M, N, K);
				}
				else
				{
					gemmProxy[select[i]].pFunImp(A, B, C1, M, N, K);
				}
			}
		}
		printf("ret %d\n",compare(C, C1, M*N));
	}

	printf("All Test Done.\n");
	int qwerdf = 0;
	scanf_s("%d", &qwerdf);
}


int main()
{
	testAll();
	return 0;
}

