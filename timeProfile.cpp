#include"timeProfile.h"

//
/// UnitTimeProfile
///
UnitTimeProfile::UnitTimeProfile(std::string description, int cnt)
{
	m_description = description;
	m_count = cnt;

	QueryPerformanceFrequency(&m_lTimeElapse);
	QueryPerformanceCounter(&m_lStart);
}

UnitTimeProfile::~UnitTimeProfile()
{
	double dTimeElapse;
	QueryPerformanceCounter(&m_lEnd);
	dTimeElapse = (double)(m_lEnd.QuadPart - m_lStart.QuadPart) / m_lTimeElapse.QuadPart;
	printf("%s: ave %8.6lf ms in %d times\n", m_description.c_str(), 1000 * dTimeElapse / m_count, m_count);
}