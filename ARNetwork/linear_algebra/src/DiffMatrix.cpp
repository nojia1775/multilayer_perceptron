#include "../include/DiffMatrix.hpp"

DiffMatrix::DiffMatrix(const size_t& dimension) : Matrix<float>(dimension, dimension), _dimension(dimension)
{
	for (size_t i = 0 ; i < _dimension ; i++)
	{
		for (size_t j = i + 1 ; j < _dimension ; j++)
		{
			if (i == j - 1)
				_matrix[i][j] = j;
		}
	}
}

DiffMatrix&	DiffMatrix::operator=(const DiffMatrix& diffMatrix)
{
	if (this != &diffMatrix)
	{
		if (_dimension != diffMatrix._dimension)
			throw Error("Error : matrices must have the same dimensions");
		_matrix = diffMatrix._matrix;
	}
	return *this;
}