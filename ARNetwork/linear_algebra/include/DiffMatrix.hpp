#pragma once

#include "LinearAlgebra.hpp"

class	DiffMatrix : public Matrix<float>
{
	private:
		const size_t		_dimension;

	public:
					DiffMatrix(const size_t& dimension);
		inline			~DiffMatrix(void) {}

		inline			DiffMatrix(const DiffMatrix& diffMatrix) : Matrix<float>(diffMatrix), _dimension(diffMatrix._dimension) {}
		
		DiffMatrix&		operator=(const DiffMatrix& diffMatrix);
		inline DiffMatrix&	operator=(const Matrix<float>& matrix) { Matrix<float>::operator=(matrix); return *this; }

		inline const size_t&	dimension(void) const { return _dimension; }
};