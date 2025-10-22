#include "../../include/Matrix.hpp"
#include "../../include/Error.hpp"

template <typename T>
template <typename U>
Matrix<T>::Matrix(const Vector<U>& vector) : _nbrLines(vector.dimension()), _nbrColumns(1)
{
	if (vector.empty())
		throw Error("Error: vector is empty");
	_matrix = std::vector<std::vector<T>>(_nbrLines, std::vector<T>(_nbrColumns));
	for (size_t i = 0 ; i < _nbrLines ; i++)
	{
		for (size_t j = 0 ; j < _nbrColumns ; j++)
		{
			if constexpr (std::is_same<U, Complex>::value)
			{
				if constexpr (std::is_same<T, Complex>::value)
					_matrix[i][j] = vector[i];
				else
					_matrix[i][j] = static_cast<float>(vector[i].getRealPart());
			}
			else
				_matrix[i][j] = static_cast<float>(vector[i]);
		}
	}
}

template <typename T>
template <typename U>
Matrix<T>::Matrix(const Matrix<U>& matrix)
{
	if (matrix.empty())
		throw Error("Error: matrix is empty");
	_nbrLines = matrix.getNbrLines();
	_nbrColumns = matrix.getNbrColumns();
	_matrix = std::vector<std::vector<T>>(_nbrLines, std::vector<T>(_nbrColumns));
	for (size_t i = 0 ; i < _nbrLines ; i++)
	{
		for (size_t j = 0 ; j < _nbrColumns ; j++)
		{
			if constexpr (std::is_same<U, Complex>::value && !std::is_same<T, Complex>::value)
				_matrix[i][j] = matrix[i][j].getRealPart();
			else
				_matrix[i][j] = matrix[i][j];
		}
	}
}

template <typename T>
template <typename U>
Matrix<T>::Matrix(const std::initializer_list<std::initializer_list<U>>& list)
{
	for (const auto data : list)
		if (data.size() != list.begin()->size())
			throw Error("Error : initializers must have the same dimensions");
	_nbrLines = list.size();
	_nbrColumns = list.begin()->size();
	_matrix = std::vector<std::vector<T>>(_nbrLines, std::vector<T>(_nbrColumns));
	_matrix.clear();
	_matrix = std::vector<std::vector<T>>(_nbrLines, std::vector<T>(_nbrColumns));
	size_t i = 0;
	for (const auto& datas : list)
	{
		size_t j = 0;
		for (const auto& data : datas)
		{
			if constexpr (std::is_same<U, Complex>::value)
			{
				if constexpr (std::is_same<T, Complex>::value)
					_matrix[i][j] = data;
				else
					_matrix[i][j] = static_cast<float>(data.getRealPart());
			}
			else
				_matrix[i][j] = static_cast<float>(data);
			j++;
		}
		i++;
	}
}

template <typename T>
Matrix<T>::Matrix(const vector2& vector)
{
	for (const auto& data : vector)
		if (data.size() != vector[0].size())
			throw Error("Error : vectors must have the same dimensions");
	_nbrLines = vector.size();
	_nbrColumns = vector[0].size();
	_matrix = std::vector<std::vector<T>>(_nbrLines, std::vector<T>(_nbrColumns));
	_matrix.clear();
	for (size_t i = 0 ; i < _nbrLines ; i++)
		_matrix.emplace_back(vector[i]);
}