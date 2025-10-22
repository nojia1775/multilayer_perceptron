#include "../include/Complex.hpp"

template <typename T>
Complex&	Complex::operator=(const T& number)
{
	_real = static_cast<float>(number);
	_imaginary = 0;
	return *this;
}