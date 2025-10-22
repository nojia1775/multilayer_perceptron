#include "../../include/Vector.hpp"
#include "../../include/Error.hpp"
#include "../../include/Matrix.hpp"

template <typename T>
template <typename U>
Vector<T>&	Vector<T>::operator=(const Vector<U>& vector)
{
	if (empty() || vector.empty())
		throw Error("Error: vector is empty");
	if (reinterpret_cast<const void *>(this) != reinterpret_cast<const void *>(&vector))
	{
		for (size_t i = 0 ; i < vector.dimension() ; i++)
		{
			if constexpr (std::is_same<U, Complex>::value)
			{
				if constexpr (std::is_same<T, Complex>::value)
					_vector[i] = vector[i];
				else
					_vector[i] = static_cast<float>(vector[i].getRealPart());
			}
			else
				_vector[i] = static_cast<float>(vector[i]);
		}
	}
	return *this;
}

template <typename T>
template <typename U>
Vector<T>	Vector<T>::operator+(const Vector<U>& vector) const
{
	if (empty() || vector.empty())
		throw Error("Error: vector is empty");
	if (dimension() != vector.dimension())
		throw Error("Error : vectors must have the same dimensions");
	Vector<T> result(dimension());
	for (size_t i = 0 ; i < vector.dimension() ; i++)
		result._vector[i] = _vector[i] + vector._vector[i];
	return result;
}

template <typename T>
template <typename U>
Vector<T>	Vector<T>::operator-(const Vector<U>& vector) const
{
	if (empty() || vector.empty())
		throw Error("Error: vector is empty");
	if (dimension() != vector.dimension())
		throw Error("Error : vectors must have the same dimensions");
	Vector<T> result(dimension());
	for (size_t i = 0 ; i < vector.dimension() ; i++)
		result._vector[i] = _vector[i] - vector._vector[i];
	return result;
}

template <typename T>
template <typename U>
Vector<T>	Vector<T>::operator*(const Vector<U>& vector) const
{
	if (empty() || vector.empty())
		throw Error("Error: vector is empty");
	if (dimension() != vector.dimension())
		throw Error("Error : vectors must have the same dimensions");
	Vector<T> result(dimension());
	for (size_t i = 0 ; i < vector.dimension() ; i++)
		result._vector[i] = _vector[i] * vector._vector[i];
	return result;
}

template <typename T>
template <typename U>
Vector<T>	Vector<T>::operator*(const U& number) const
{
	if (empty())
		throw Error("Error: vector is empty");
	Vector<T> result(dimension());
	for (size_t i = 0 ; i < dimension() ; i++)
		result._vector[i] = _vector[i] * number;
	return result;
}

template <typename T>
T&	Vector<T>::operator[](const size_t& index)
{
	if (empty())
		throw Error("Error: vector is empty");
	if (index > _vector.size() - 1)
		throw Error("Error : index out of range");
	return _vector[index];
}

template <typename T>
const T&	Vector<T>::operator[](const size_t& index) const
{
	if (empty())
		throw Error("Error: vector is empty");
	if (index > _vector.size() - 1)
		throw Error("Error : index out of range");
	return _vector[index];
}

template <typename T>
template <typename U>
bool	Vector<T>::operator==(const Vector<U>& vector) const
{
	if (empty() || vector.empty())
		throw Error("Error: vector is empty");
	if (vector.dimension() != dimension())
		return false;
	for (size_t i = 0 ; i < dimension() ; i++)
		if (_vector[i] != vector._vector[i])
			return false;
	return true;
}

template <typename T>
template <typename U>
bool	Vector<T>::operator!=(const Vector<U>& vector) const
{
	if (empty() || vector.empty())
		throw Error("Error: vector is empty");
	if (vector.dimension() != dimension())
		return true;
	for (size_t i = 0 ; i < dimension() ; i++)
		if (_vector[i] != vector._vector[i])
			return true;
	return false;
}

template <typename T>
template <typename U>
Vector<T>&	Vector<T>::operator=(const Matrix<U>& matrix)
{
	if (matrix.empty())
		throw Error("Error: matrix is empty");
	if (matrix.getNbrColumns() != 1)
		throw Error("Error : the number of column of the matrix must be 1");
	_vector = std::vector<T>(matrix.getNbrLines());
	for (size_t i = 0 ; i < matrix.getNbrLines() ; i++)
		_vector[i] = matrix[i][0];
	return *this;
}