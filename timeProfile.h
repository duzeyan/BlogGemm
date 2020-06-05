#ifndef _TIME_PROFILE_H_
#define _TIME_PROFILE_H_

#include<Windows.h>
#include<string>
#include<random>
#include<vector>
#include<map>
#include<functional>

using std::vector;
using std::string;
using std::map;
using std::function;

class UnitTimeProfile
{
public:
	UnitTimeProfile(std::string description, int cnt);
	~UnitTimeProfile();
private:
	int m_count;
	std::string m_description;
	LARGE_INTEGER  m_lStart;
	LARGE_INTEGER  m_lEnd;
	LARGE_INTEGER  m_lTimeElapse;
};

#endif