#pragma once
#include "defs.h"

class SeparableSurrogateFunctional
{
public:
	float m_b;
	size_t m_epocs;
	float m_s;
	float m_lambda;
	float m_c;

private:
	bool _forward();
	bool _backward();
	bool _shrink();
	float _rho();
	float _func_obj();
	bool _search_line();
	bool _sesop();


public:
	SeparableSurrogateFunctional();
	~SeparableSurrogateFunctional();
	bool set_param();
	bool execute();
};