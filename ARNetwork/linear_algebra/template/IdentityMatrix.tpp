#include "../include/IdentityMatrix.hpp"

template <typename T>
IdentityMatrix<T>::IdentityMatrix(const size_t& dimension) : Matrix<T>(dimension, dimension), _dimension(dimension)
{
	for (size_t i = 0 ; i < _dimension ; i++)
	{
		for (size_t j = 0 ; j < _dimension ; j++)
		{
			if (i == j)
				this->_matrix[i][j] = 1;
			else
				this->_matrix[i][j] = 0;
		}
	}
}

template <typename T>
IdentityMatrix<T>&	IdentityMatrix<T>::operator=(const IdentityMatrix& identityMatrix)
{
	if (this != &identityMatrix)
	{
		if (_dimension == identityMatrix._dimension)
			throw Error("Error : matrices must have the same dimensions");
		this->_matrix = identityMatrix._matrix;
	}
	return *this;
}