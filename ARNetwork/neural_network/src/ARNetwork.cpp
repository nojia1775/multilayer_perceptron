#include "../include/ARNetwork.hpp"

/**
 * @brief Construct a neural network based on a vector
 * 
 * Construct a neural network based on a vector
 * Each index is a layer and the value is the number of neurals in this layer
 * 
 * @param network:
 * @param network[0] number of inputs
 * @param network[n-1] number of outputs
 * @param everything between corresponds to the hidden layers and neurals
 */
ARNetwork::ARNetwork(const std::vector<size_t>& network)
{
	if (network.size() < 2)
		throw Error("Error: not enough neurals in the network");
	for (const auto& neurals : network)
		if (neurals == 0)
			throw Error("Error: number of neurals can't be 0");
	size_t inputs = network[0];
	size_t outputs = network[network.size() - 1];
	size_t hidden_layers = network.size() - 2;
	_weights = std::vector<Matrix<double>>(hidden_layers + 1);
	_bias = std::vector<Vector<double>>(hidden_layers + 1);
	_inputs = Vector<double>(inputs);
	_outputs = Vector<double>(outputs);
	_z = std::vector<Vector<double>>(hidden_layers + 1);
	_a = std::vector<Vector<double>>(hidden_layers + 1);
	_learning_rate = 0.1;
	for (size_t i = 0 ; i < hidden_layers + 1 ; i++)
	{
		_weights[i] = Matrix<double>(network[i + 1], network[i]);
		_bias[i] = Vector<double>(network[i + 1]);
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
			_bias[i][j] = random_double(-1, 1);
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
		{
			for (size_t k = 0 ; k < _weights[i].getNbrColumns() ; k++)
				_weights[i][j][k] = random_double(-1, 1);
		}
	}
}

ARNetwork::ARNetwork(const ARNetwork& arn) : _inputs(arn._inputs), _outputs(arn._outputs), _weights(arn._weights), _z(arn._z), _a(arn._a), _bias(arn._bias), _learning_rate(arn._learning_rate) {}

ARNetwork	ARNetwork::operator=(const ARNetwork& arn)
{
	if (this != &arn)
	{
		_inputs = arn._inputs;
		_outputs = arn._outputs;
		_weights = arn._weights;
		_z = arn._z;
		_a = arn._a;
		_bias = arn._bias;
		_learning_rate = arn._learning_rate;
		_layer_function = arn._layer_function;
		_output_function = arn._output_function;
		_loss_function = arn._loss_function;
	}
	return *this;
}

const double&	ARNetwork::get_bias(const size_t& i, const size_t& j) const
{
	if (i > _bias.size() - 1)
		throw Error("Error: index out of range");
	if (j < _bias[i].dimension() - 1)
		throw Error("Error: index out of range");
	return _bias[i][j];
}

void	ARNetwork::set_bias(const size_t& i, const size_t& j, const double& bias)
{
	if (i > _bias.size() - 1)
		throw Error("Error: index out of range");
	if (j < _bias[i].dimension() - 1)
		throw Error("Error: index out of range");
	_bias[i][j] = bias;
}

/**
 * @brief Perform a forward pass through the neural network
 * 
 * @param inputs vector which contains the values to compute
 * @param layer_functions name of the activation function for the hidden layers
 * @param output_functions name of the activation function for the output layer
 * 
 * @return vector which contains the outputs
 */
Vector<double>	ARNetwork::feed_forward(const Vector<double>& inputs, const std::string& layer_functions, const std::string& output_functions)
{
	auto output_activation = ActivationFactory::create(output_functions);
	auto layer_activation = ActivationFactory::create(layer_functions);
	set_inputs(inputs);
	Matrix<double> neurals = _inputs;
	_a[0] = _inputs;
	for (size_t i = 0 ; i < nbr_hidden_layers() + 1 ; i++)
	{
		_z[i] = _weights[i] * neurals + Matrix<double>(_bias[i]);
		neurals = _z[i];
		try
		{
			if (i == nbr_hidden_layers())
				neurals = output_activation->activate_vector(neurals);
			else
				neurals = layer_activation->activate_vector(neurals);
		}
		catch (...)
		{
			for (size_t j = 0 ; j < neurals.getNbrLines() ; j++)
			{
				if (i == 0)
					neurals[j][0] = output_activation->activate_scalar(neurals[j][0]);
				else
					neurals[j][0] = layer_activation->activate_scalar(neurals[j][0]);
			}
		}
		if (i != nbr_hidden_layers())
			_a[i + 1] = neurals;
	}
	_outputs = Vector<double>(neurals);
	return _outputs;
}

/**
 * @brief Perform backward pass through the neural network
 * 
 * Using of the gradient regression to change the weights and bias based on their gradient
 * 
 * @param dW vector of matrices which contains the sum of the weights' gradient
 * @param dZ vector of matrices which contains the sum of the z value's gradient
 * @param loss_functions name of the loss functions used to compute the gradient of the network's outputs' gradient
 * @param layer_functions name of the activation function used to compute hidden layers' weights' gradient
 * @param output_functions name of the activation function used to compute the gradient of the outputs' z value
 * @param y vector which contains the value we want to reach with the neural network
 */
void	ARNetwork::back_propagation(std::vector<Matrix<double>>& dW, std::vector<Matrix<double>>& dZ, const std::string& loss_functions, const std::string& layer_functions, const std::string& output_functions, const Vector<double>& y)
{
	auto output_activation = ActivationFactory::create(output_functions);
	auto layer_activation = ActivationFactory::create(layer_functions);
	auto loss_activation = LossFactory::create(loss_functions);
	Matrix<double> dA(loss_activation->derive(_outputs, y));
	for (int l = nbr_hidden_layers() ; l >= 0 ; l--)
	{
		Vector<double> tmp(_z[l].dimension());
		try
		{
			if (l == (int)nbr_hidden_layers())
				tmp = output_activation->derive_vector(tmp);
			else
				tmp = layer_activation->derive_vector(tmp);
		}
		catch (...)
		{
			for (size_t i = 0 ; i < tmp.dimension() ; i++)
			{
				if (l == (int)nbr_hidden_layers())
					tmp[i] = output_activation->derive_scalar(tmp[i]);
				else
					tmp[i] = layer_activation->derive_scalar(tmp[i]);
			}
		}
		Matrix<double> z = dA.hadamard(tmp);
		Matrix<double> w = z * Matrix<double>(_a[l]).transpose();
		dZ[l] = dZ[l] + z;
		dW[l] = dW[l] + w;
		dA = _weights[l].transpose() * z;
	}
}

void	ARNetwork::update_weights_bias(const std::vector<Matrix<double>>& dW, const std::vector<Matrix<double>>& dZ, const size_t& batch)
{
	for (size_t layer = 0 ; layer < nbr_hidden_layers() + 1 ; layer++)
	{
		_weights[layer] = _weights[layer] - dW[layer] * _learning_rate * static_cast<double>(1.0 / static_cast<double>(batch));
		_bias[layer] = Matrix<double>(_bias[layer]) - dZ[layer] * _learning_rate * static_cast<double>(1.0 / static_cast<double>(batch));
	}
}

static void	valid_lists(const std::vector<std::vector<std::vector<double>>>& inputs, const std::vector<std::vector<std::vector<double>>>& outputs, const size_t& size_inputs, const size_t& size_outputs)
{
	if (inputs.size() != outputs.size())
		throw Error("Error: the number of batch of inputs and outputs must be the same");
	for (size_t i = 0 ; i < inputs.size() ; i++)
	{
		if (inputs[i].size() != outputs[i].size())
			throw Error("Error: batch of inputs and outputs have different size");
		for (size_t j = 0 ; j < inputs[i].size() ; j++)
		{
			if (inputs[i][j].size() != size_inputs)
				throw Error("Error: example " + j + std::string(" must have " + size_inputs + std::string(" inputs")));
			if (outputs[i][j].size() != size_outputs)
				throw Error("Error: example " + j + std::string(" must have " + size_inputs + std::string(" outputs")));
		}
	}
}

void	ARNetwork::process(const batch_type& inputs, const batch_type& outputs, const double& sstot, std::map<size_t, std::pair<double, double>>& track_training, const size_t& epoch, const bool& back)
{
	double nbr_vectorial_inputs = 0;
	for (const auto& batch : inputs)
		nbr_vectorial_inputs += batch.size();
	double ssres = 0;
	double loss_index = 0;
	auto loss_activation = LossFactory::create(_loss_function);
	for (size_t j = 0 ; j < inputs.size() ; j++)
	{
		std::vector<Matrix<double>> dW(nbr_hidden_layers() + 1);
		std::vector<Matrix<double>> dZ(nbr_hidden_layers() + 1);
		for (size_t k = 0 ; k < inputs[j].size() ; k++)
		{
			Vector<double> prediction = feed_forward(inputs[j][k], _layer_function, _output_function);
			loss_index += loss_activation->activate(prediction, outputs[j][k]);
			for (size_t l = 0 ; l < prediction.dimension() ; l++)
				ssres += pow(prediction[l] - outputs[j][k][l], 2);
			if (back)
				back_propagation(dW, dZ, _loss_function, _layer_function, _output_function, outputs[j][k]);
		}
		if (back)
			update_weights_bias(dW, dZ, inputs[j].size());
	}
	double r2 = 1.0 - ssres / sstot;
	track_training[epoch] = {loss_index / nbr_vectorial_inputs, r2};
}

static double	compute_sstot(const std::vector<std::vector<std::vector<double>>>& outputs)
{
	double count_outputs = 0;
	double nbr_scalar_outputs = 0;
	for (const auto& batch : outputs)
	{
		for (const auto& sample : batch)
		{
			for (const auto& coef : sample)
			{
				nbr_scalar_outputs += coef;
				count_outputs++;
			}
		}
	}
	double mean_output = nbr_scalar_outputs / count_outputs;
	double sstot = 0;
	for (const auto& batch : outputs)
		for (const auto& sample : batch)
			for (const auto& coef : sample)
				sstot += pow(coef - mean_output, 2);
	return sstot;
}

/**
 * @brief Train the neural network based on a data set
 * 
 * @param loss_functions name of the loss function
 * @param layer_functions name of the activation function used in the hidden layers
 * @param output_functions name of the activation function used in the output layer
 * @param inputs batches of inputs
 * @param outputs batches of outputs we want to reach
 * @param epochs number of epoch 
 *
 * @return A pair of map which contains a pair containing the loss and r2 for each epoch
 */
std::pair<std::map<size_t, std::pair<double, double>>, std::map<size_t, std::pair<double, double>>>	ARNetwork::train(const std::string& loss_functions, const std::string& layer_functions, const std::string& output_functions, const std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>>& inputs, const std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<std::vector<double>>>>& outputs, const size_t& epochs)
{
	if (inputs.first.empty() || inputs.second.empty())
		throw Error("Error: train or validation inputs are missing");
	if (outputs.first.empty() || outputs.second.empty())
		throw Error("Error: train or validation inputs are missing");
	auto loss_activation = LossFactory::create(loss_functions);
	_loss_function = loss_functions;
	_layer_function = layer_functions;
	_output_function = output_functions;
	valid_lists(inputs.first, outputs.first, size_inputs(), size_outputs());
	valid_lists(inputs.second, outputs.second, size_inputs(), size_outputs());
	model_measures_type track_training;
	for (size_t i = 0 ; i < epochs ; i++)
	{
		process(inputs.first, outputs.first, compute_sstot(outputs.first), track_training.first, i, true);
		process(inputs.second, outputs.second, compute_sstot(outputs.second), track_training.second, i, false);
	}
	return track_training;
}

/**
 * @brief transform a list of inputs into a list of group of @param batch numbers
 * 
 * @param list list of different inputs
 * @param batch size of the groups we want to create
 * 
 * @return a list of groups of inputs
 */
std::vector<std::vector<std::vector<double>>>	ARNetwork::batching(const std::vector<std::vector<double>>& list, const size_t& batch)
{
	if (batch == 0)
		throw Error("Error: batch cannot be 0");
	size_t groups = batch > list.size() ? 1 : (size_t)(list.size() / batch);
	if (batch < list.size())
		groups += list.size() % batch == 0 ? 0 : 1;
	batch_type result(groups);
	size_t index = 0;
	for (size_t i = 0 ; i < list.size() ; i++)
	{
		if (i != 0 && i % batch == 0)
			index++;
		result[index].push_back(list[i]);
	}
	return result;
}

void	ARNetwork::randomize_weights(const double& min, const double& max)
{
	for (size_t i = 0 ; i < nbr_hidden_layers() + 1 ; i++)
	{
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
		{
			for (size_t k = 0 ; k < _weights[i].getNbrColumns() ; k++)
				_weights[i][j][k] = random_double(min, max);
		}
	}
}

void	ARNetwork::randomize_weights(const size_t& layer, const double& min, const double& max)
{
	if (layer > nbr_hidden_layers())
		throw Error("Error: index out of range");
	for (size_t i = 0 ; i < _weights[layer].getNbrLines() ; i++)
	{
		for (size_t j = 0 ; j < _weights[layer].getNbrColumns() ; j++)
			_weights[layer][i][j] = random_double(min, max);
	}
}

void	ARNetwork::randomize_bias(const double& min, const double& max)
{
	size_t i;
	for (i = 0 ; i < nbr_hidden_layers() ; i++)
	{
		for (size_t j = 0 ; j < nbr_hidden_neurals(i) ; j++)
			_bias[i][j] = random_double(min, max);
	}
	for (size_t j = 0 ; j < size_outputs() ; j++)
		_bias[i][j] = random_double(min, max);
}

void	ARNetwork::randomize_bias(const size_t& layer, const double& min, const double& max)
{
	if (layer > nbr_hidden_layers())
		throw Error("Error: index out of range");
	for (size_t j = 0 ; j < nbr_hidden_neurals(layer) ; j++)
		_bias[layer][j] = random_double(min, max);
}