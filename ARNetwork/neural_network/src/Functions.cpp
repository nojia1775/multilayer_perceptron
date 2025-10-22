#include "../include/Functions.hpp"

Vector<double>	SoftMax::activate_vector(const Vector<double>& x) const
{
	Vector<double> output(x.dimension());
	double maxVal = *std::max_element(x.getStdVector().begin(), x.getStdVector().end());
	double sumExp = 0.0;
	for (size_t i = 0; i < x.dimension(); ++i)
	{
		output[i] = std::exp(x[i] - maxVal);
		sumExp += output[i];
	}
	for (size_t i = 0; i < output.dimension(); ++i)
		output[i] /= sumExp;
	return output;
}

Matrix<double>	SoftMax::derive_vector(const Vector<double>& x) const
{
	Vector<double> s = activate_vector(x);
    	size_t n = s.dimension();
    	Matrix<double> jacobian(n, n);

    	for (size_t i = 0; i < n; ++i)
    	{
    	    for (size_t j = 0; j < n; ++j)
    	    {
    	        if (i == j)
    	            jacobian[i][j] = s[i] * (1.0 - s[j]);
    	        else
    	            jacobian[i][j] = -s[i] * s[j];
    	    }
    	}
    	return jacobian;
}

double	MSE::activate(const Vector<double>& a, const Vector<double>& b) const
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	if (a.dimension() != b.dimension())
		throw Error("Error: vectors must have the same dimension");
	double sum = 0;
	for (size_t i = 0 ; i < a.dimension() ; i++)
		sum += pow(a[i] - b[i], 2);
	return sum / (2.0 * static_cast<double>(a.dimension()));
}

Matrix<double>	MSE::derive(const Vector<double>& a, const Vector<double>& b) const
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	if (a.dimension() != b.dimension())
		throw Error("Error: vectors must have the same dimension");
	Matrix<double> gradients(a.dimension(), 1);
	for (size_t i = 0 ; i < gradients.getNbrLines() ; i++)
		gradients[i][0] = (a[i] - b[i]) / a.dimension();
	return gradients;
}

double	BCE::activate(const Vector<double>& a, const Vector<double>& b) const
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	if (a.dimension() != b.dimension())
		throw Error("Error: vectors must have the same dimension");
	double sum = 0;
	for (size_t i = 0 ; i < b.dimension() ; i++)
	{
		double y_hat;
		if (a[i] < std::numeric_limits<double>::epsilon())
			y_hat = std::numeric_limits<double>::epsilon();
		else if (a[i] > 1 - std::numeric_limits<double>::epsilon())
			y_hat = 1 - std::numeric_limits<double>::epsilon();
		else 
			y_hat = a[i];
		sum += b[i] * std::log(y_hat) + (1 - b[i]) * std::log(1 - y_hat);
	}
	return -sum / static_cast<double>(a.dimension());
}

Matrix<double>	BCE::derive(const Vector<double>& a, const Vector<double>& b) const
{
	if (a.empty() || b.empty())
		throw Error("Error: vector is empty");
	if (a.dimension() != b.dimension())
		throw Error("Error: vectors must have the same dimension");
	Matrix<double> gradients(a.dimension(), 1);
	for (size_t i = 0 ; i < a.dimension() ; i++)
	{
		double y_hat;
		if (a[i] < std::numeric_limits<double>::epsilon())
			y_hat = std::numeric_limits<double>::epsilon();
		else if (a[i] > 1 - std::numeric_limits<double>::epsilon())
			y_hat = 1 - std::numeric_limits<double>::epsilon();
		else 
			y_hat = a[i];
		gradients[i][0] = (y_hat - b[i]) / (y_hat * (1.0 - y_hat) * a.dimension());
	}
	return gradients;
}

std::unique_ptr<IActivation>	ActivationFactory::create(const std::string& function)
{
	if (function == "relu") return std::make_unique<ReLU>();
	if (function == "sigmoid") return std::make_unique<Sigmoid>();
	if (function == "tanh") return std::make_unique<TanH>();
	if (function == "leakyrelu") return std::make_unique<LeakyReLU>();
	if (function == "identity") return std::make_unique<Identity>();
	if (function == "softmax") return std::make_unique<SoftMax>();
	throw Error("Error: unknown activation function: " + function);
}

std::unique_ptr<ILoss>	LossFactory::create(const std::string& function)
{
	if (function == "mse") return std::make_unique<MSE>();
	if (function == "bce") return std::make_unique<BCE>();
	throw Error("Error: unknown loss function: " + function);
}