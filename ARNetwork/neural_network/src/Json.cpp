#include "../include/ARNetwork.hpp"

/**
 * @brief Create a json file which contains the bias, weights, loss function, layer function and output function of the neural network
 * 
 * @param file_name name of the json file
 */
void	ARNetwork::get_json(const std::string& file_name) const
{
	nlohmann::json data;
	data["weights"] = nlohmann::json::array();
	for (size_t i = 0 ; i < nbr_hidden_layers() + 1 ; i++)
	{
		nlohmann::json matrix = nlohmann::json::array();
		for (size_t j = 0 ; j < _weights[i].getNbrLines() ; j++)
			matrix.push_back(_weights[i].getLine(j).getStdVector());
		data["weights"].push_back(matrix);
	}
	data["bias"] = nlohmann::json::array();
	for (size_t i = 0 ; i < nbr_hidden_layers() + 1 ; i++)
		data["bias"].push_back(_bias[i].getStdVector());
	data["learning_rate"] = _learning_rate;
	data["hidden_activation"] = _hidden_activation;
	data["output_activation"] = _output_activation;
	data["loss"] = _loss;
	std::ofstream file(file_name);
	if (file.is_open())
	{
		file << data.dump();
		std::cout << "Log saved in " << file_name << "\n";
		file.close();
	}
	else
		std::cerr << "Error: could't save log\n";
}

ARNetwork::ARNetwork(const std::string& file_name)
{
	std::ifstream file(file_name);
	if (!file.is_open())
	{
		std::cout << "Impossible to open " << file_name << "\n";
		return;
	}
	nlohmann::json data;
	try { file >> data; }
	catch (const nlohmann::json::parse_error& e) { std::cout << e.what() << "\n"; }
	_inputs = Vector<double>(data["weights"][0][0].size());
	_outputs = Vector<double>(data["weights"][data["weights"].size() - 1].size());
	_weights = std::vector<Matrix<double>>(data["weights"].size());
	_bias = std::vector<Vector<double>>(data["bias"].size());
	_z = std::vector<Vector<double>>(data["weights"].size());
	_a = std::vector<Vector<double>>(data["weights"].size());
	_learning_rate = data["learning_rate"];
	for (size_t layer = 0 ; layer < data["weights"].size() ; layer++)
	{
		_weights[layer] = Matrix<double>(data["weights"][layer].size(), data["weights"][layer][0].size());
		_bias[layer] = Vector<double>(data["bias"][layer].size());
	}
	for (size_t layer = 0 ; layer < data["weights"].size() ; layer++)
	{
		for (size_t row = 0 ; row < data["weights"][layer].size() ; row++)
		{
			_bias[layer][row] = data["bias"][layer][row];
			for (size_t col = 0 ; col < data["weights"][layer][row].size() ; col++)
				_weights[layer][row][col] = data["weights"][layer][row][col];
		}
	}
}