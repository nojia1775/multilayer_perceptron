#pragma once

#include "Complex.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "Error.hpp"
#include "IdentityMatrix.hpp"
#include "DiffMatrix.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <variant>
#include <cmath>

template <typename T>
Vector<T>	linear_combination(const std::vector<Vector<T>>& vectors, const std::vector<T>& scalars)
{
	if (vectors.size() != scalars.size())
		throw Error("Error: lists must have the same dimension");
	for (const auto& vector : vectors)
	{
		if (vector.dimension() != vectors[0].dimension())
			throw Error("Error: vectors must have the same dimensions");
	}
	std::vector<Vector<T>> finalVectors;
	for (size_t i = 0 ; i < vectors.size() ; i++)
		finalVectors.push_back(vectors[i] * scalars[i]);
	Vector<T> result(vectors[0].dimension());
	for (size_t i = 0 ; i < vectors.size() ; i++)
		result = result + finalVectors[i];
	return result;
}

template <typename Test, template <typename...> class Ref>
struct is_specialization_of : std::false_type {};

template <template <typename...> class Ref, typename... Args>
struct is_specialization_of<Ref<Args...>, Ref> : std::true_type {};


template <typename T>
T	lerp(const T& a, const T& b, const float& cursor)
{
	if constexpr (is_specialization_of<T, Matrix>::value)
	{
		if (a.empty() || b.empty())
			throw Error("Error: matrix is empty");
		if (a.getNbrLines() != b.getNbrLines() || a.getNbrColumns() != b.getNbrColumns())
			throw Error("Error: matrices must have the same dimensions");
	}
	else if constexpr (is_specialization_of<T, Vector>::value)
	{
		if (a.empty() || b.empty())
			throw Error("Error: vector is empty");
		if (a.dimension() != b.dimension())
			throw Error("Error: vectors must have the same dimension");
	}
	if (cursor < 0 || cursor > 1)
		throw Error("Error: cursor must be between 0 and 1 included");
	return a + (b - a) * cursor;
}