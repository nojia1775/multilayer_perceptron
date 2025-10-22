#include "../include/Complex.hpp"
#include "../include/Error.hpp"

Complex&	Complex::operator=(const Complex& complex)
{
	if (this != &complex)
	{
		_real = complex._real;
		_imaginary = complex._imaginary;
	}
	return *this;
}

Complex	Complex::operator*(const Complex& complex) const
{
	Complex result;
	result._imaginary = _real * complex._imaginary + _imaginary * complex._real;
	result._real = _real * complex._real - _imaginary * complex._imaginary;
	return result;
}

Complex	Complex::operator/(const Complex& complex) const
{
	if (complex == 0)
		throw Error("Error : division by 0 is undefined");
	Complex a, b;
	a = *this * complex.getConjugate();
	b = complex * complex.getConjugate();
	Complex result(a / b.getRealPart());
	return result;
}

Complex	Complex::operator/(const float& number) const
{
	if (number == 0)
		throw Error("Error : division by 0 is undefined");
	Complex result(_real / number, _imaginary / number);
	return result;
}

void	Complex::operator*=(const Complex& complex)
{
	float real = _real;
	float imaginary = _imaginary;
	_real = real * complex._real - imaginary * complex._imaginary;
	_imaginary = real * complex._imaginary + imaginary * complex._real;
}

void	Complex::operator/=(const Complex& complex)
{
	Complex tmp(*this / complex);
	_real = tmp._real;
	_imaginary = tmp._imaginary;
}

void	Complex::operator/=(const float& number)
{
	Complex tmp(*this / number);
	_real = tmp._real;
	_imaginary = tmp._imaginary;
}

std::ostream&	operator<<(std::ostream& o, const Complex& complex)
{
	if (complex == 0)
	{
		std::cout << "0";
		return o;
	}
	if (complex.getRealPart())
	{
		o << complex.getRealPart();
		if (complex.getImaginaryPart() > 0)
			o << " + " << complex.getImaginaryPart() << "i";
		else if (complex.getImaginaryPart() < 0)
			o << " - " << -complex.getImaginaryPart() << "i"; 
	}
	else
	{
		if (complex.getImaginaryPart() > 0)
			o << complex.getImaginaryPart() << "i";
		else if (complex.getImaginaryPart() < 0)
			o << complex.getImaginaryPart() << "i";
	}
	return o;
}

float	Complex::getArgument(void) const
{
	if (_real == 0)
		return 1e10;
	return std::atan(_imaginary / _real);
}