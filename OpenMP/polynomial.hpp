#pragma once

#ifndef MOD
#define MOD 10
#endif

#include <array>
using namespace std;

template <typename ring>
class Polynomial {
public:
	array<ring, DEG> coeff = {};

	Polynomial()
	{
		for (int i = 0; i < DEG; ++i) coeff[i] = (ring) (rand() % MOD);
	}

	Polynomial(const ring& ref)
	{
		coeff[0] = ref;
	}

	Polynomial(const array<ring, DEG>& _coeff) : coeff(_coeff)
	{
		assert(DEG == _coeff.size());
	}

	ring& at(const int& i) const
	{ 
		assert(0 <= i && i < DEG);
		return coeff[i];
	}

	const Polynomial operator+(const Polynomial& ref) const
	{
		array<ring, DEG> res = {};

		for (int i = 0; i < DEG; ++i)
		{
			res[i] = coeff[i]+ref.coeff[i];
		}

		return Polynomial(res);
	}

	const Polynomial& operator+=(const Polynomial& ref)
	{
		for (int i = 0; i < DEG; ++i)
		{
			coeff[i] += ref.coeff[i];
		}

		return *this;
	}

	const Polynomial operator-(const Polynomial& ref) const
	{
		array<ring, DEG> res = {};

		for (int i = 0; i < DEG; ++i)
		{
			res[i] = coeff[i]-ref.coeff[i];
		}

		return Polynomial(res);
	}

	const Polynomial operator*(const Polynomial& ref) const
	{
		array<ring, DEG> res = {};

		for (int i = 0; i < DEG; ++i)
		{
			for (int j = 0; j < DEG; ++j)
			{
				res[(i+j)%DEG] += coeff[i]*ref.coeff[j];
			}
		}

		return Polynomial(res);
	}

	const Polynomial operator*(const ring& ref) const
	{
		array<ring, DEG> res = {};

		for (int i = 0; i < DEG; ++i)
		{
			for (int j = 0; j < DEG; ++j)
			{
				res[(i+j)%DEG] += coeff[i]*ref;
			}
		}

		return Polynomial(res);
	}

	const Polynomial operator*=(const Polynomial& ref)
	{
		array<ring, DEG> res = {};

		for (int i = 0; i < DEG; ++i)
		{
			for (int j = 0; j < DEG; ++j)
			{
				res[(i+j)%DEG] += coeff[i]*ref.coeff[j];
			}
		}

		coeff = res;
		return *this;
	}

	const Polynomial& operator=(const ring& ref)
	{
		fill(coeff.begin(), coeff.end(), 0);
		coeff[0] = ref;
		return *this;
	}

	const Polynomial& operator=(const Polynomial& ref)
	{
		coeff = ref.coeff;
		return *this;
	}

	const bool operator==(const Polynomial& ref)
	{
		for (int i = 0; i < DEG; ++i)
		{
			if (coeff[i] != ref.coeff[i]) return false;
		}
		return true;
	}

	const bool operator!=(const Polynomial& ref)
	{
		return !(*this == ref);
	}
};
