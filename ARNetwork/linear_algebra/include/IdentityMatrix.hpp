#pragma once

#include "LinearAlgebra.hpp"

template <typename T>
class	IdentityMatrix : public Matrix<T>
{
	private:
		const size_t		_dimension;

	public:
					IdentityMatrix(const size_t& dimension);
		inline			~IdentityMatrix(void) {}

		inline			IdentityMatrix(const IdentityMatrix<T>& identityMatrix) : Matrix<float>(identityMatrix), _dimension(identityMatrix._dimension) {}
		IdentityMatrix<T>&		operator=(const IdentityMatrix<T>& IdentityMatrix);

		inline const size_t&	dimension(void) const { return _dimension; }
};

#include "../template/IdentityMatrix.tpp"