#pragma once

#include <iostream>
#include <cmath>
#include <type_traits>

class Error;

class	Complex
{
	public:
		float		_real;
		float		_imaginary;

	public:
				Complex(void) : _real(0), _imaginary(0) {}
				~Complex(void) {}

				Complex(const float& real, const float& imaginary) : _real(real), _imaginary(imaginary) {}
				Complex(const Complex& complex) : _real(complex._real), _imaginary(complex._imaginary) {}
				template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
				Complex(const T& number) : _real(static_cast<float>(number)), _imaginary(0) {}

		const float&	getImaginaryPart(void) const { return _imaginary; }
		const float&	getRealPart(void) const { return _real; }
		Complex		getConjugate(void) const { return Complex(_real, -_imaginary); }
		float		getModule(void) const { return sqrt(pow(_real, 2) + pow(_imaginary, 2)); }
		float		getArgument(void) const;

		Complex&	operator=(const Complex& complex);
				template <typename T>
		Complex&	operator=(const T& number);
		Complex		operator+(const Complex& complex) const { return Complex(_real + complex._real, _imaginary + complex._imaginary); }
		Complex		operator+(const float& number) const { return Complex(_real + number, _imaginary); }
		Complex		operator-(const Complex& complex) const { return Complex(_real - complex._real, _imaginary - complex._imaginary); }
		Complex		operator-(const float& number) const { return Complex(_real - number, _imaginary); }
		Complex		operator/(const Complex& complex) const;
		Complex		operator/(const float& number) const;
		Complex		operator*(const Complex& complex) const;
		Complex		operator*(const float& number) const { return Complex(_real * number, _imaginary * number); }
		void		operator+=(const Complex& complex) { _real += complex._real; _imaginary += complex._imaginary; }
		void		operator+=(const float& number) { _real += number; }
		void		operator-=(const Complex& complex) { _real -= complex._real; _imaginary -= complex._imaginary; }
		void		operator-=(const float& number) { _real -= number; }
		void		operator*=(const Complex& complex);
		void		operator*=(const float& number) { _real *= number; _imaginary *= number; }
		void		operator/=(const Complex& complex);
		void		operator/=(const float& number);
		bool		operator==(const Complex& complex) const { return _imaginary == complex._imaginary && _real == complex._real; }
				template <typename T>
		bool		operator==(const T& number) const { return _real == number && _imaginary == 0; }
		bool		operator!=(const Complex& complex) const { return _imaginary != complex._imaginary || _real != complex._real; }
				template <typename T>
		bool		operator!=(const T& number) const { return _real != number || _imaginary != 0; }
				template <typename T>
		bool		operator>(const T& number) const { return _real > number; }
				template <typename T>
		bool		operator<(const T& number) const { return _real < number; }
};

std::ostream&	operator<<(std::ostream& o, const Complex& complex);

#include "../template/Complex.tpp"