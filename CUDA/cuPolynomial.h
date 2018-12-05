#ifndef POLYNOMIAL_H_
#define POLYNOMIAL_H_


////////////////////////////////////////////////////////////////////////////////
// poly4
////////////////////////////////////////////////////////////////////////////////

typedef float4 cuFloatPoly4;
static __inline__ __host__ __device__
cuFloatPoly4 make_cuFloatPoly4(const float& x, const float& y, const float& z, const float& w)
{
	return make_float4(x, y, z, w);
}

typedef double4 cuDoublePoly4;
static __inline__ __host__ __device__
cuDoublePoly4 make_cuDoublePoly4(const double& x, const double& y, const double& z, const double& w)
{
	return make_double4(x, y, z, w);
}


////////////////////////////////////////////////////////////////////////////////
// add
////////////////////////////////////////////////////////////////////////////////

static __inline__ __host__ __device__
cuFloatPoly4 operator+(const cuFloatPoly4& a, const cuFloatPoly4& b)
{
	return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

static __inline__ __host__ __device__
void operator+=(cuFloatPoly4& a, const cuFloatPoly4& b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

static __inline__ __host__ __device__
cuDoublePoly4 operator+(const cuDoublePoly4& a, const cuDoublePoly4& b)
{
	return make_double4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

static __inline__ __host__ __device__
void operator+=(cuDoublePoly4& a, const cuDoublePoly4& b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}


////////////////////////////////////////////////////////////////////////////////
// sub
////////////////////////////////////////////////////////////////////////////////

static __inline__ __host__ __device__
cuFloatPoly4 operator-(const cuFloatPoly4& a, const cuFloatPoly4& b)
{
	return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

static __inline__ __host__ __device__
cuDoublePoly4 operator-(const cuDoublePoly4& a, const cuDoublePoly4& b)
{
	return make_double4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}


////////////////////////////////////////////////////////////////////////////////
// mul
////////////////////////////////////////////////////////////////////////////////

static __inline__ __host__ __device__
cuFloatPoly4 operator*(const cuFloatPoly4& a, const cuFloatPoly4& b)
{
	float c0 = a.x*b.x + a.y*b.w + a.z*b.z + a.w*b.y;
	float c1 = a.x*b.y + a.y*b.x + a.z*b.w + a.w*b.z;
	float c2 = a.x*b.z + a.y*b.y + a.z*b.x + a.w*b.w;
	float c3 = a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x;
	return make_float4(c0, c1, c2, c3);
}

static __inline__ __host__ __device__
cuFloatPoly4 operator*(const float& a, const cuFloatPoly4& b)
{
	return make_float4(a*b.x, a*b.y, a*b.z, a*b.w);
}

static __inline__ __host__ __device__
cuFloatPoly4 operator*(const cuFloatPoly4& a, const float& b)
{
	return make_float4(a.x*b, a.y*b, a.z*b, a.w*b);
}

static __inline__ __host__ __device__
cuDoublePoly4 operator*(const cuDoublePoly4& a, const cuDoublePoly4& b)
{
	double c0 = a.x*b.x + a.y*b.w + a.z*b.z + a.w*b.y;
	double c1 = a.x*b.y + a.y*b.x + a.z*b.w + a.w*b.z;
	double c2 = a.x*b.z + a.y*b.y + a.z*b.x + a.w*b.w;
	double c3 = a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x;
	return make_double4(c0, c1, c2, c3);
}

static __inline__ __host__ __device__
cuDoublePoly4 operator*(const double& a, const cuDoublePoly4& b)
{
	return make_double4(a*b.x, a*b.y, a*b.z, a*b.w);
}

static __inline__ __host__ __device__
cuDoublePoly4 operator*(const cuDoublePoly4& a, const double& b)
{
	return make_double4(a.x*b, a.y*b, a.z*b, a.w*b);
}


////////////////////////////////////////////////////////////////////////////////
// etc
////////////////////////////////////////////////////////////////////////////////

static __inline__ __host__ __device__
float LinfDistPoly4(const cuFloatPoly4& a)
{
	return (a.x > 0 ? a.x : -a.x)
	     + (a.y > 0 ? a.y : -a.y)
	     + (a.z > 0 ? a.z : -a.z)
	     + (a.w > 0 ? a.w : -a.w);
}

static __inline__ __host__ __device__
double LinfDistPoly4(const cuDoublePoly4& a)
{
	return (a.x > 0 ? a.x : -a.x)
	     + (a.y > 0 ? a.y : -a.y)
	     + (a.z > 0 ? a.z : -a.z)
	     + (a.w > 0 ? a.w : -a.w);
}


#endif