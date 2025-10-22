#include "../../include/Vector.hpp"
#include "../../include/Error.hpp"
#include "../../include/Matrix.hpp"

template <typename T>
void	Vector<T>::display(void) const
{
	std::cout << "[";
	size_t i = 0;
	for (const auto& data : _vector)
	{
		if (i < _vector.size() - 1)
			std::cout << data << " , ";
		else
			std::cout << data;
		i++;
	}
	std::cout << "]\n";
}

template <typename T, typename Ta, typename Tb>
T	dotProduct(const Vector<Ta>& a, const Vector<Tb>& b)
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	if (a.dimension() != b.dimension())
		throw Error("Error : vectors must have the same dimension");
	if constexpr (std::is_same<Ta, Complex>::value || std::is_same<Tb, Complex>::value)
	{
		Complex result;
		for (size_t i = 0 ; i < a.dimension() ; i++)
		result += Complex(a[i]) * Complex(b[i]);
		return result;
	}
	T result{};
	for (size_t i = 0 ; i < a.dimension() ; i++)
		result += a[i] * b[i];
	return result;
}

template <typename T>
T	dot(const Vector<T>& a, const Vector<T>& b)
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	if (a.dimension() != b.dimension())
		throw Error("Error: vectors must have the same dimension");
	T result{};
	for (size_t i = 0 ; i < a.dimension() ; i++)
		result += a[i] * b[i];
	return result;
}

template <typename T>
void	Vector<T>::normalise(void)
{
	if (empty())
		throw Error("Error: vector is empty");
	for (auto& data : _vector)
		data /= norm();
}

static inline Vector<Complex>	computeProj(const Vector<Complex>& e, const Vector<Complex>& vector) { return Vector<Complex>(e * dotProduct<Complex>(vector, e)); }

template <typename T>
std::vector<Vector<Complex>>	orthonormalize(const std::vector<Vector<T>>& vectors)
{
	for (const auto& vector : vectors)
	{
		if (vector.empty())
			throw Error("Error: vector is empty");
		if (vector.dimension() != vectors[0].dimension())
			throw Error("Error : vectors must have the same dimensions");
	}
	for (size_t i = 0 ; i < vectors.size() ; i++)
	{
		for (size_t j = 0 ; j < vectors.size() ; j++)
		{
			if (i != j)
				if (linearlyDependants(vectors[i], vectors[j]))
					throw Error("Error : some vectors are dependant");
		}
	}
	std::vector<Vector<Complex>> newVectors(vectors.size());
	std::vector<Vector<Complex>> result;
	for (size_t i = 0 ; i < vectors.size() ; i++)
		newVectors[i] = Vector<Complex>(vectors[i]);
	result.push_back(newVectors[0].normalised());
	for (size_t i = 1 ; i < newVectors.size() ; i++)
	{
		for (size_t j = 0 ; j < result.size() ; j++)
			newVectors[i] = newVectors[i] - computeProj(result[j], newVectors[i]);
		result.push_back(newVectors[i].normalised());
	}
	return result;
}

template <typename A, typename B>
bool	linearlyDependants(const Vector<A>& a, const Vector<B>& b)
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	if (a.dimension() != b.dimension())
		throw Error("Error : vectors must have the same dimensions");
	Complex scalar = 0;
	for (size_t i = 0 ; i < a.dimension() ; i++)
	{
		b[i] == 0 ? scalar = 0 : scalar = Complex(a[i]) / Complex(b[i]);
		if (scalar != 0)
			break;
	}
	if (scalar == 0)
		throw Error("Error : at least one vector is null");
	for (size_t i = 0 ; i < a.dimension() ; i++)
	{
		if (Complex(a[i]) / Complex(b[i]) != scalar)
			return false;
	}
	return true;
}

template <typename T>
float	Vector<T>::norm_1(void) const
{
	if (empty())
		throw Error("Error: vector is empty");
	float result = 0;
	Vector<Complex> vector(*this);
	for (size_t i = 0 ; i < dimension() ; i++)
		result += vector[i].getModule();
	return result;
}

template <typename T>
float	Vector<T>::norm_inf(void) const
{
	if (empty())
		throw Error("Error: vector is empty");
	float result = 0;
	Vector<Complex> vector(*this);
	for (size_t i = 0 ; i < dimension() ; i++)
		result = vector[i].getModule() > result ? vector[i].getModule() : result;
	return result;
}

template <typename T>
float	angle_cos(const Vector<T>& a, const Vector<T>& b)
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	Complex result = dot(a, b) / (a.norm() * b.norm());
	return result.getRealPart();
}

template <typename T>
Vector<T>	cross_product(const Vector<T>& a, const Vector<T>& b)
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	if (a.dimension() != 3 || b.dimension() != 3)
		throw Error("Error: vectors' dimension must be 3");
	return Vector<T>({a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]});
}

template <typename T>
float	Vector<T>::norm(void) const
{
	if (empty())
		throw Error("Error: vector is empty");
	Vector<Complex> Cvector = *this;
	float norm = 0;
	for (size_t i = 0 ; i < dimension() ; i++)
		norm += pow(Cvector[i].getModule(), 2);
	return sqrt(norm);
}

template <typename T>
Vector<T>	Vector<T>::normalised(void) const
{
	if (empty())
		throw Error("Error: vector is empty");
	Vector<T> result(dimension());
	for (size_t i = 0 ; i < dimension() ; i++)
		result[i] = _vector[i] / norm();
	return result;
}

template <typename T>
template <typename F>
Vector<T>	Vector<T>::apply(F f) const
{
	if (f == NULL)
		return *this;
	Vector<T> result(*this);
	for (size_t i = 0 ; i < result.dimension() ; i++)
		result[i] = f(result[i]);
	return result;
}

template <typename T>
Vector<T>	Vector<T>::hadamard(const Vector<T>& vector) const
{
	if (dimension() != vector.dimension())
		throw Error("Error: vectors must be the same dimension");
	Vector<T> result(*this);
	for (size_t i = 0 ; i < result.dimension() ; i++)
		result[i] *= vector[i];
	return result;
}