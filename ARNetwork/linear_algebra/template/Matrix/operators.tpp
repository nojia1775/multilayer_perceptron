#include "../../include/Matrix.hpp"
#include "../../include/Error.hpp"

template <typename T>
template <typename U>
Matrix<T>&	Matrix<T>::operator=(const Matrix<U>& matrix)
{
	if (empty())
		throw Error("Error: matrix is empty");
	if (reinterpret_cast<const void *>(this) != reinterpret_cast<const void *>(&matrix))
	{
		_nbrLines = matrix.getNbrLines();
		_nbrColumns = matrix.getNbrColumns();
		_matrix = std::vector<std::vector<T>>(matrix.getNbrLine(), std::vector<T>(matrix.getNbrColumns()));
		for (size_t i = 0 ; i < _nbrLines ; i++)
		{
			for (size_t j = 0 ; j < _nbrColumns ; j++)
			{
				if constexpr (std::is_same<U, Complex>::value)
				{
					if constexpr (std::is_same<T, Complex>::value)
						_matrix[i][j] = matrix[i][j];
					else
						_matrix[i][j] = static_cast<float>(matrix[i][j].getRealPart());
				}
				else
					_matrix[i][j] = static_cast<float>(matrix[i][j]);
			}
		}
	}
	return *this;
}

template <typename T>
template <typename U>
Matrix<T>&	Matrix<T>::operator=(const std::initializer_list<std::initializer_list<U>>& list)
{
	if (empty())
		throw Error("Error: matrix is empty");
	for (const auto data : list)
		if (data.size() != list[0].size())
			throw Error("Error : initializers must have the same dimensions");
	_nbrLines = list.size();
	_nbrColumns = list.begin()->size();
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
	return *this;
}

template <typename T>
template <typename U>
Matrix<T>&	Matrix<T>::operator=(const std::vector<std::vector<U>>& vector)
{
	if (empty())
		throw Error("Error: matrix is empty");
	for (const auto data : vector)
		if (data.size() != vector[0].size())
			throw Error("Error : vectors must have the same dimensions");
	_nbrLines = vector.size();
	_nbrColumns = vector[0].size();
	_matrix = std::vector<std::vector<T>>(_nbrLines, std::vector<T>(_nbrColumns));
	for (size_t i = 0 ; i < _nbrLines ; i++)
	{
		for (size_t j = 0 ; j < _nbrColumns ; j++)
		{
			if constexpr (std::is_same<U, Complex>::value)
			{
				if constexpr (std::is_same<T, Complex>::value)
					_matrix[i][j] = vector[i][j];
				else
					_matrix[i][j] = static_cast<float>(vector[i][j].getRealPart());
			}
			else
				_matrix[i][j] = static_cast<float>(vector[i][j]);
		}
	}
	return *this;
}

template <typename T>
template <typename U>
Matrix<T>&	Matrix<T>::operator=(const Vector<U>& vector)
{
	if (vector.empty())
		throw Error("Error: vector is empty");
	_nbrLines = vector.dimension();
	_nbrColumns = 1;
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
	return *this;
}

template <typename T>
std::vector<T>&	Matrix<T>::operator[](const size_t& index)
{
	if (empty())
		throw Error("Error: matrix is empty");
	if (index > _nbrLines - 1)
		throw Error("Error : index out of range");
	return _matrix[index];
}

template <typename T>
const std::vector<T>&	Matrix<T>::operator[](const size_t& index) const
{
	if (empty())
		throw Error("Error: matrix is empty");
	if (index > _nbrLines - 1)
		throw Error("Error : index out of range");
	return _matrix[index];
}

template <typename T>
template <typename U>
Matrix<T>	Matrix<T>::operator*(const Matrix<U>& matrix) const
{
	if (empty() || matrix.empty())
		throw Error("Error: matrix is empty");
	Matrix<T> result(_nbrLines, matrix.getNbrColumns());
	for (size_t i = 0 ; i < result._nbrLines ; i++)
	{
		for (size_t j = 0 ; j < result._nbrColumns ; j++)
			result[i][j] = dotProduct<T>(getLine(i), matrix.getColumn(j));
	}
	return result;
}

template <typename T>
Matrix<T>&	Matrix<T>::operator*=(const float& number)
{
	if (empty())
		throw Error("Error: matrix is empty");
	for (auto& datas : _matrix)
	{
		for (auto& data : datas)
			data *= number;
	}
	return *this;
}

template <typename T>
Matrix<T>	Matrix<T>::operator*(const float& number) const
{
	if (empty())
		throw Error("Error: matrix is empty 9");
	Matrix result(_nbrLines, _nbrColumns);
	for (size_t i = 0 ; i < _nbrLines ; i++)
	{
		for (size_t j = 0 ; j < _nbrColumns ; j++)
			result[i][j] = _matrix[i][j] * number;
	}
	return result;
}

template <typename T>
Matrix<T>	Matrix<T>::operator/(const float& number) const
{
	if (empty())
		throw Error("Error: matrix is empty");
	if (number == 0)
		throw Error("Error : division by 0 is undefined");
	return Matrix<T>(*this) * (1 / number);
}

template <typename T>
template <typename U>
Matrix<U>	Matrix<T>::operator*(const Vector<U>& vector) const
{
	if (empty())
		throw Error("Error: matrix is empty");
	if (vector.empty())
		throw Error("Error: vector is empty");
	// if (_nbrColumns != vector.dimension())
	// 	throw Error("Error : vector.dimension must be equal to matrice.column");
	return Matrix<U>(*this * Matrix<U>(vector));
}

template <typename T>
template <typename U>
bool	Matrix<T>::operator==(const Matrix<U>& matrix) const
{
	if (empty() || matrix.empty())
		throw Error("Error: matrix is empty");
	if (_nbrLines != matrix.getNbrLines() || _nbrColumns != matrix.getNbrColumns())
		return false;
	for (size_t i = 0 ; i < getNbrLines() ; i++)
	{
		for (size_t j = 0 ; j < getNbrColumns() ; j++)
		{
			if (_matrix[i][j] != matrix[i][j])
				return false;
		}
	}
	return true;
}

template <typename T>
template <typename U>
Matrix<T>	Matrix<T>::operator+(const Matrix<U>& matrix) const
{
	if (matrix.empty())
		throw Error("Error: matrix is empty");
	if (empty())
		return matrix;
	if (_nbrColumns != matrix.getNbrColumns() || _nbrLines != matrix.getNbrLines())
		throw Error("Error : matrices must have the same dimensions");
	Matrix<T> result(_nbrLines, matrix.getNbrColumns());
	for (size_t i = 0 ; i < result._nbrLines ; i++)
	{
		for (size_t j = 0 ; j < result._nbrColumns ; j++)
			result[i][j] = _matrix[i][j] + matrix[i][j];
	}
	return result;
}

template <typename T>
template <typename U>
Matrix<T>	Matrix<T>::operator-(const Matrix<U>& matrix) const
{
	if (matrix.empty())
		throw Error("Error: matrix is empty");
	if (empty())
		return matrix * -1;
	if (_nbrColumns != matrix.getNbrColumns() || _nbrLines != matrix.getNbrLines())
		throw Error("Error : matrices must have the same dimensions");
	Matrix<T> result(_nbrLines, matrix.getNbrColumns());
	for (size_t i = 0 ; i < result._nbrLines ; i++)
	{
		for (size_t j = 0 ; j < result._nbrColumns ; j++)
			result[i][j] = _matrix[i][j] - matrix[i][j];
	}
	return result;
}

template <typename T>
Matrix<Complex>	Matrix<T>::operator*(const Complex& complex) const
{
	if (empty())
		throw Error("Error: matrix is empty");
	Matrix<Complex> result(_nbrLines, _nbrColumns);
	for (size_t i = 0 ; i < _nbrLines ; i++)
	{
		for (size_t j = 0 ; j < _nbrColumns ; j++)
			result[i][j] = complex * _matrix[i][j];
	}
	return result;
}