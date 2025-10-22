#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include "../../linear_algebra/include/LinearAlgebra.hpp"

class	IActivation
{
	public:
		virtual			~IActivation(void) {}
		virtual std::string	name(void) const = 0;
		virtual	double		activate_scalar(const double& x) const { (void)x; throw Error("Error: function is not scalar based"); }
		virtual	double		derive_scalar(const double& x) const { (void)x; throw Error("Error: function is not scalar based"); }
		virtual	Vector<double>	activate_vector(const Vector<double>& vector) const { (void)vector; throw Error("Error: function is not vector-based"); }
		virtual	Matrix<double>	derive_vector(const Vector<double>& vector) const { (void)vector; throw Error("Error: function is not vector-based"); }
};

class	ReLU : public IActivation
{
	std::string	name(void) const override { return "relu"; }
	double		activate_scalar(const double& x) const override { return x <= 0 ? 0 : x; }
	double		derive_scalar(const double& x) const override { return x <= 0 ? 0 : 1; }
};

class	Sigmoid : public IActivation
{
	std::string	name(void) const override { return "sigmoid"; }
	double		activate_scalar(const double& x) const override { return 1 / (1 + exp(-x)); }
	double		derive_scalar(const double& x) const override { return activate_scalar(x) * (1 - activate_scalar(x)); }
};

class	TanH : public IActivation
{
	std::string	name(void) const override { return "tanh"; }
	double		activate_scalar(const double& x) const override { return std::tanh(x); }
	double		derive_scalar(const double& x) const override { return 1 - std::tanh(x) * std::tanh(x); }
};

class	LeakyReLU : public IActivation
{
	std::string	name(void) const override { return "leakyrelu"; }
	double		activate_scalar(const double& x) const override { return x <= 0 ? x * 0.01 : x; }
	double		derive_scalar(const double& x) const override { return x <= 0 ? 0.01 : 1; }
};

class	Identity : public IActivation
{
	std::string	name(void) const override { return "identity"; }
	double		activate_scalar(const double& x) const override { return x; }
	double		derive_scalar(const double& x) const override { return x >= 0 ? 1 : -1; }
};

class	SoftMax : public IActivation
{
	std::string	name(void) const override { return "softmax"; }
	Vector<double>	activate_vector(const Vector<double>& x) const override;
	Matrix<double>	derive_vector(const Vector<double>& x) const override;
};

class	ILoss
{
	public:
		virtual			~ILoss(void) {}
		virtual	std::string	name(void) const = 0;
		virtual double		activate(const Vector<double>& a, const Vector<double>& b) const = 0;
		virtual Matrix<double>	derive(const Vector<double>& a, const Vector<double>& b) const = 0;
};

class	MSE : public ILoss
{
	std::string	name(void) const override { return "mse"; }
	double		activate(const Vector<double>& a, const Vector<double>& b) const override;
	Matrix<double>	derive(const Vector<double>& a, const Vector<double>& b) const override;
};

class	BCE : public ILoss
{
	std::string	name(void) const override { return "bce"; }
	double		activate(const Vector<double>& a, const Vector<double>& b) const override;
	Matrix<double>	derive(const Vector<double>& a, const Vector<double>& b) const override;
};

class	ActivationFactory
{
	public:
		static std::unique_ptr<IActivation>	create(const std::string& function);
};

class	LossFactory
{
	public:
		static std::unique_ptr<ILoss>	create(const std::string& function);
};