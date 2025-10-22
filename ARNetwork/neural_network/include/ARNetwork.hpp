#pragma once

#include "../../linear_algebra/include/LinearAlgebra.hpp"
#include "Json.hpp"
#include "Functions.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>

class	ARNetwork
{
	typedef std::vector<std::vector<std::vector<double>>> batch_type;

	private:
		Vector<double>					_inputs;
		Vector<double>					_outputs;
		std::vector<Matrix<double>>			_weights;
		std::vector<Vector<double>>			_z;
		std::vector<Vector<double>>			_a;
		std::vector<Vector<double>>			_bias;
		double						_learning_rate;
		std::string					_hidden_activation;
		std::string					_output_activation;
		std::string					_loss;

	public:
								ARNetwork(const std::vector<size_t>& network);
								ARNetwork(const std::string& file_name);
								~ARNetwork(void) {}

								ARNetwork(const ARNetwork& arn);
		ARNetwork					operator=(const ARNetwork& arn);

		const Vector<double>&				get_inputs(void) const { return _inputs; }
		const double&					get_input(const size_t& index) { if (index > _inputs.dimension() - 1) throw Error("Error: out of range"); else return _inputs[index]; }
		const std::vector<Matrix<double>>&		get_weights(void) const { return _weights; }
		const Matrix<double>&				get_weights(const size_t& layer) const { if (layer > _weights.size() - 1) throw Error("Error: index out of range"); else return _weights[layer]; }
		const std::vector<Vector<double>>&		get_bias(void) const { return _bias; }
		const Vector<double>&				get_bias(const size_t& index) const { if (index > _bias.size() - 1) throw Error("Error: index out of range"); else return _bias[index]; }
		const double&					get_bias(const size_t& i, const size_t& j) const;
		const double&					get_learning_rate(void) const { return _learning_rate; }
		const Vector<double>&				get_outputs(void) const { return _outputs; }
		const double&					get_output(const size_t& index) { if (index > _outputs.dimension() - 1) throw Error("Error: index out of range"); else return _outputs[index]; }
		void						get_json(const std::string& file_name) const;

		size_t						nbr_inputs(void) const { return _inputs.dimension(); }
		size_t						nbr_hidden_layers(void) const { return _weights.size() - 1; }
		size_t						nbr_hidden_neurals(const size_t& layer) const { if (layer > nbr_hidden_layers() - 1) throw Error("Error: index out of range"); return _weights[layer].getNbrLines(); }
		size_t						nbr_bias(void) const { return _bias.size(); }
		size_t						nbr_outputs(void) const { return _outputs.dimension(); }

		void						set_inputs(const Vector<double>& inputs) { _inputs = inputs; }
		void						set_weights(std::vector<Matrix<double>>& weights) { _weights = weights; }
		void						set_weights(const size_t& index, const Matrix<double>& weights) { if (index > _weights.size() - 1) throw Error("Error: index out of range"); else _weights[index] = weights; }
		void						set_bias(const std::vector<Vector<double>>& bias) { _bias = bias; }
		void						set_bias(const size_t& index, const Vector<double>& bias) { if (index > _bias.size() - 1) throw Error("Error: index out of range"); else _bias[index] = bias; }
		void						set_bias(const size_t& i, const size_t& j, const double& bias);
		void						set_learning_rate(const double& learning_rate) { _learning_rate = learning_rate; }

		Vector<double>					feed_forward(const Vector<double>& inputs, const std::string& layer_functions, const std::string& output_functions);
		void						back_propagation(std::vector<Matrix<double>>& dW, std::vector<Matrix<double>>& dZ, const std::string& loss_functions,const std::string& layer_functions,const std::string& output_functions, const Vector<double>& y);
		std::map<std::string, std::vector<double>>	train(const std::string& loss_functions, const std::string& layer_functions, const std::string& output_functions, const batch_type& inputs, const batch_type& outputs, const size_t& epochs);
		void						update_weights_bias(const std::vector<Matrix<double>>& dW, const std::vector<Matrix<double>>& dZ, const size_t& batch);
		static batch_type				batching(const std::vector<std::vector<double>>& list, const size_t& batch);
		void						randomize_weights(const size_t& layer, const double& min, const double& max);
		void						randomize_weights(const double& min, const double& max);
		void						randomize_bias(const size_t& layer, const double& min, const double& max);
		void						randomize_bias(const double& min, const double& max);
};

inline std::mt19937&	global_urng(void)
{
	std::random_device rd;
	static std::mt19937 gen(rd());
	return gen;
}

inline double	random_double(const double& min, const double& max)
{
	std::uniform_real_distribution<double> dist(min, max);
	return dist(global_urng());
}