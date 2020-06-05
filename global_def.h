#ifndef _GLOBAL_DEF_H_
#define _GLOBAL_DEF_H_

typedef void(*GEMM_FUN)(const float *A, const float *B, float *C, int M, int N, int K);

#define _STR(x) #x
#define _DECLARE_GEMM_IMP(f,isT) \
	GemmImp(f, _STR(f), isT)
class GemmImp
{
public:
	GemmImp(GEMM_FUN pFunImp, const char *pStrDescription, int transType)
		:pFunImp(pFunImp), pStrDescription(pStrDescription), transType(transType){}

	GEMM_FUN pFunImp;
	const char *pStrDescription;
	int transType;//0:no transpose; 1:mat trasnpose;2:vector4 transpose
};

#endif