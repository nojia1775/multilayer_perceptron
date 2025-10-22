#include "../../include/Vector.hpp"
#include "../../include/Error.hpp"
#include "../../include/Matrix.hpp"

template <typename T>
template <typename U>
Vector<T>::Vector(const Vector<U>& vector)
{
	if (vector.empty())
		throw Error("Error: vector is empty");
	_vector = std::vector<T>(vector.dimension());
	for (size_t i = 0 ; i < vector.dimension() ; i++)
	{
		if constexpr (std::is_same<U, Complex>::value)
		{
			if constexpr (std::is_same<T, Complex>::value)
				_vector[i] = vector[i];
			else
				_vector[i] = vector[i].getRealPart();
		}
		else
			_vector[i] = static_cast<float>(vector[i]);
	}
}

template <typename T>
template <typename U>
Vector<T>::Vector(const Matrix<U>& matrix)
{
	if (matrix.empty())
		throw Error("Error: matrix is empty");
	if (matrix.getNbrColumns() != 1)
		throw Error("Error : matrix must have only 1 column");
	_vector = std::vector<T>(matrix.getNbrLines());
	*this = matrix.getColumn(0);
}

template <typename T>
template <typename U>
Vector<T>::Vector(const std::initializer_list<U>& list)
{
	for (const auto& value : list)
		_vector.push_back(static_cast<T>(value));
}