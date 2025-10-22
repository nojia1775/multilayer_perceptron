#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include "Complex.hpp"
#include "LinearAlgebra.hpp"
#include <variant>

template <typename T>
class Matrix;

template <typename T>
class	Vector
{
	private:
		std::vector<T>		_vector;

	public:
					Vector(void) : _vector(std::vector<T>()) {}
					~Vector(void) {}
					template <typename U>
					Vector(const Vector<U>& vector);
					template <typename U>
					Vector(const std::vector<U>& stdvector) : _vector(stdvector) {}
					Vector(const size_t& dimension) : _vector(std::vector<T>(dimension)) {}
					template <typename U>
					Vector(const std::initializer_list<U>& list);
					template <typename U>
					Vector(const Matrix<U>& matrix);

					template <typename U>
		Vector<T>&		operator=(const Vector<U>& vector);
					template <typename U>
		Vector<T>&		operator=(const std::vector<U>& stdvector) { _vector = stdvector; return *this; }
					template <typename U>
		Vector<T>&		operator=(const std::initializer_list<U>& list) { _vector = list; return *this; }
					template <typename U>
		Vector<T>&		operator=(const Matrix<U>& matrix);
					template <typename U>
		Vector<T>		operator+(const Vector<U>& vector) const;
					template <typename U>
		Vector<T>		operator-(const Vector<U>& vector) const;
					template <typename U>
		Vector<T>		operator*(const Vector<U>& vector) const;
					template <typename U>
		Vector<T>		operator*(const U& number) const;
		T&			operator[](const size_t& index);
		const T&		operator[](const size_t& index) const;
					template <typename U>
		bool			operator==(const Vector<U>& vector) const;
					template <typename U>
		bool			operator!=(const Vector<U>& vector) const;

		size_t			dimension(void) const { return _vector.size(); }
		float			norm(void) const;
		Vector<T>		normalised(void) const;
		const std::vector<T>&	getStdVector(void) const { return _vector; }

		void			display(void) const;
		void			normalise(void);
		bool			empty(void) const { return _vector.size() == 0; }
		float			norm_1(void) const;
		float			norm_inf(void) const;
		template <typename F>
		Vector<T>		apply(F f) const;
		Vector<T>		hadamard(const Vector<T>& vector) const;
};

template <typename T, typename Ta, typename Tb>
T	dotProduct(const Vector<Ta>& a, const Vector<Tb>& b);

template <typename Ta, typename Tb>
inline bool	areOrthogonals(const Vector<Ta>& a, const Vector<Tb>& b) { return dotProduct<Complex>(a, b) > -1e-5 && dotProduct<Complex>(a, b) < 1e-5; }

template <typename T>
std::vector<Vector<Complex>>	orthonormalize(const std::vector<Vector<T>>& vectors);

template <typename A, typename B>
bool	linearlyDependants(const Vector<A>& a, const Vector<B>& b);

// MATRIX 42

template <typename T>
T	dot(const Vector<T>& a, const Vector<T>& b);

template <typename T>
float	angle_cos(const Vector<T>& a, const Vector<T>& b);

template <typename T>
Vector<T>	cross_product(const Vector<T>& a, const Vector<T>& b);

#include "../template/Vector/constructors.tpp"
#include "../template/Vector/operators.tpp"
#include "../template/Vector/functions.tpp"