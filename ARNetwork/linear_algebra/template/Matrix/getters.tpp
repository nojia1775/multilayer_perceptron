#include "../../include/Matrix.hpp"
#include "../../include/Error.hpp"

template <typename T>
Vector<T>	Matrix<T>::getLine(const size_t& index) const
{
	if (empty())
		throw Error("Error: matrix is empty");
	if (index > getNbrLines() - 1)
		throw Error("Error : index out of range");
	return Vector<T>(_matrix[index]);
}

template <typename T>
Vector<T>	Matrix<T>::getColumn(const size_t& index) const
{
	if (empty())
		throw Error("Error: matrix is empty");
	if (index > getNbrColumns() - 1)
		throw Error("Error : index out of range");
	Vector<T> result(getNbrLines());
	for (size_t i = 0 ; i < getNbrLines() ; i++)
		result[i] = _matrix[i][index];
	return result;
}