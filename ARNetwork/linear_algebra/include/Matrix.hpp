#pragma once

#include <iostream>
#include "Vector.hpp"
#include "Complex.hpp"
#include "IdentityMatrix.hpp"

class Error;

template <typename T>
class	Matrix
{
	typedef std::vector<std::vector<T>> vector2;

	protected:
		vector2				_matrix;
		size_t				_nbrLines;
		size_t				_nbrColumns;
	
	public:
		virtual				~Matrix(void) {}
						Matrix(void) : _matrix(), _nbrLines(0), _nbrColumns(0) {}

						Matrix(const size_t& nbrLines, const size_t& nbrColumns) : _matrix(vector2(nbrLines, std::vector<T>(nbrColumns, T{}))), _nbrLines(nbrLines), _nbrColumns(nbrColumns) {}
						template <typename U>
						Matrix(const std::initializer_list<std::initializer_list<U>>& list);
						Matrix(const vector2& vector);
						template <typename U>
						Matrix(const Vector<U>& vector);
						template <typename U>
						Matrix(const Matrix<U>& matrix);

						template <typename U>
		Matrix<T>&			operator=(const Matrix<U>& matrix);
						template <typename U>
		Matrix<T>&			operator=(const std::initializer_list<std::initializer_list<U>>& list);
						template <typename U>
		Matrix<T>&			operator=(const std::vector<std::vector<U>>& vector);
						template <typename U>
		Matrix<T>&			operator=(const Vector<U>& vector);
		std::vector<T>&			operator[](const size_t& index);
		const std::vector<T>&		operator[](const size_t& index) const;
						template <typename U>
		Matrix<T>			operator*(const Matrix<U>& matrix) const;
						template <typename U>
		Matrix<T>			operator+(const Matrix<U>& matrix) const;
						template <typename U>
		Matrix<T>			operator-(const Matrix<U>& matrix) const;
						template <typename U>
		Matrix<T>			operator-(const U& number) const { return Matrix<T>(*this - IdentityMatrix<T>(getNbrColumns()) * number); }
		Matrix<T>			operator*(const float& number) const;
		Matrix<T>			operator/(const float& number) const;
						template <typename U>
		Matrix<U>			operator*(const Vector<U>& vector) const;
		Matrix<Complex>			operator*(const Complex& complex) const;
		Matrix<T>&			operator*=(const float& number);
						template <typename U>
		bool				operator==(const Matrix<U>& matrix) const;
						template <typename U>
		bool				operator!=(const Matrix<U>& matrix) const { return !operator==(matrix); }

		const size_t&			getNbrLines(void) const { return _nbrLines; }
		const size_t&			getNbrColumns(void) const { return _nbrColumns; }
		Vector<T>			getLine(const size_t& index) const;
		Vector<T>			getColumn(const size_t& index) const;
		T				determinant(void) const;
		Matrix<T>			inverse(void) const;
		Matrix<T>			comatrix(void) const;
		Matrix<T>			transpose(void) const;
		Matrix<T>			adjugate(void) const { return comatrix().transpose(); }
		std::vector<Complex>		eigenValues(void) const;
		std::vector<Vector<Complex>>	eigenVectors(void) const;

		void				display(void) const;
		bool				square(void) const { return _nbrLines == _nbrColumns; }
		bool				diagonal(void) const;
		bool				empty(void) const { return _nbrColumns == 0 && _nbrLines == 0; }
		bool				inversible(void) const { return (determinant() > std::numeric_limits<float>::epsilon() || determinant() < -std::numeric_limits<float>::epsilon()) && !null(); }
		std::vector<Matrix<T>>		decompLU(size_t& swap) const;
		void				switchLine(const size_t& l1, const size_t& l2);
		void				switchColumn(const size_t& c1, const size_t& c2);
		void				switchLinePartial(size_t i, size_t j, size_t upto);
		bool				upperTriangle(void) const;
		bool				lowerTriangle(void) const;
		bool				null(void) const;
		std::vector<Matrix<Complex>>	QR(void) const;
		T				trace(void) const;
		Matrix<T>			row_echelon(void) const;
		size_t				rank(void) const;
		Matrix<T>			sumCols(void) const;
		Matrix<T>			sumLines(void) const;
						template <typename F>
		Matrix<T>			apply(F f) const;
		Matrix<T>			hadamard(const Matrix<T>& matrix) const;
};

template <typename T>
Matrix<T>	powMatrix(const Matrix<T>& matrix, const size_t& power);

#include "../template/Matrix/functions.tpp"
#include "../template/Matrix/getters.tpp"
#include "../template/Matrix/operators.tpp"
#include "../template/Matrix/constructors.tpp"