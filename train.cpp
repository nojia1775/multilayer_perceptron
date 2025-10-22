#include "ARNetwork/neural_network/include/ARNetwork.hpp"

static bool	valid_line(const std::string& line, const size_t& pos)
{
	size_t comma = 0;
	if (pos == 0)
	{
		if (line != "km,price")
			return false;
		return true;
	}
	for (size_t i = 0 ; i < line.size() ; i++)
	{
		if (!std::isdigit(line[i]) && line[i] != ',')
			return false;
		else if (line[i] == ',' && comma)
			return false;
		else if (line[i] == ',' && !std::isdigit(line[i + 1]))
			return false;
		else if (line[i] == ',')
			comma++;
	}
	if (comma == 0)
		return false;
	return true;
}

static std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>	extract_data(const std::string& file)
{
	std::ifstream data(file);
	if (!data)
		throw Error("Error: couldn't open " + file);
	std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> set;
	std::string line;
	size_t count = 0;
	while (getline(data, line))
	{
		if (valid_line(line, count) == false)
		{
			data.close();
			throw Error("Error: " + file + " is corrupted");
		}
		if (count != 0)
		{
			size_t pos = line.find(',');
			set.first.push_back({std::atof(line.substr(0, pos).c_str())});
			set.second.push_back({std::atof(line.c_str() + pos + 1)});
		}
		count++;
	}
	data.close();
	return set;
}

static std::vector<size_t>	get_network(const std::string& arg)
{
	std::vector<size_t> layers;
	layers.push_back(30);
	for (size_t i = 0 ; arg[i] ; i++)
	{
		if (!isdigit(arg[i]) && arg[i] != ' ')
			throw Error("Error: wrong format of layer");
		if (isdigit(arg[i]))
			layers.push_back(std::atoi(arg.c_str() + i));
	}
	layers.push_back(2);
	return layers;
}

static ARNetwork	parse_args(int argc, char **argv, std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>& data, std::string& layer_function, int& epoch, int& batch)
{
	if (argc == 1)
		throw Error("Error: ./train --dataset <dataset> --layer '<layers>' [--epoch <epoch> --learning_rate <learning_rate> --layer_activation <layer_activation> --batch <batch>]");
	bool datafile = false;
	double learning_rate = 0.1;
	std::vector<size_t> network;
	for (size_t i = 1 ; (int)i < argc && argv[i] ; i += 2)
	{
		if (std::string(argv[i]) == "--dataset")
		{
			if (!argv[i + 1])
				throw Error("Error: ./train --dataset <dataset> [--epoch <epoch> --learning_rate <learning_rate> --loss_function <loss_function> --layer_activation <layer_activation> --output_activation <output_activation> --batch <batch>]");
			data = extract_data(argv[i + 1]);
			datafile = true;
		}
		else if (std::string(argv[i]) == "--epoch")
		{
			if (!argv[i + 1])
				throw Error("Error: ./train --dataset <dataset> [--epoch <epoch> --learning_rate <learning_rate> --loss_function <loss_function> --layer_activation <layer_activation> --output_activation <output_activation> --batch <batch>]");
			int value;
			try { value = std::stoi(argv[i + 1]); }
			catch (...) { throw Error("Error: epoch must be a non null positive integer"); }
			if (value <= 0)
				throw Error("Error: epoch must be a non null positive integer");
			epoch = value;
		}
		else if (std::string(argv[i]) == "--learning_rate")
		{
			if (!argv[i + 1])
				throw Error("Error: ./train --dataset <dataset> [--epoch <epoch> --learning_rate <learning_rate> --loss_function <loss_function> --layer_activation <layer_activation> --output_activation <output_activation> --batch <batch>]");
			double value;
			try { value = std::stod(argv[i + 1]); }
			catch (...) { throw Error("Error: learning rate must be a non null positive double"); }
			if (value <= 0)
				throw Error("Error: learning rate must be a non null positive double");
			learning_rate = value;
		}
		else if (std::string(argv[i]) == "--layer_function")
		{
			if (!argv[i + 1])
				throw Error("Error: ./train --dataset <dataset> [--epoch <epoch> --learning_rate <learning_rate> --loss_function <loss_function> --layer_activation <layer_activation> --output_activation <output_activation> --batch <batch>]");
			layer_function = argv[i + 1];
		}
		else if (std::string(argv[i]) == "--batch")
		{
			if (!argv[i + 1])
				throw Error("Error: ./train --dataset <dataset> [--epoch <epoch> --learning_rate <learning_rate> --loss_function <loss_function> --layer_activation <layer_activation> --output_activation <output_activation> --batch <batch>]");
			double value;
			try { value = std::stoi(argv[i + 1]); }
			catch (...) { throw Error("Error: batch must be a non null positive integer"); }
			if (value <= 0)
				throw Error("Error: batch must be a non null positive integer");
			batch = value;
		}
		else if (std::string(argv[i]) == "--layer")
		{
			if (!argv[i + 1])
				throw Error("Error: ./train --dataset <dataset> [--epoch <epoch> --learning_rate <learning_rate> --loss_function <loss_function> --layer_activation <layer_activation> --output_activation <output_activation> --batch <batch>]");
			network = get_network(argv[i + 1]);
		}
		else
			throw Error("Error: unknown flag: " + std::string(argv[i]));
		if (datafile == false)
			throw Error("Error: dataset is missing");
	}
	ARNetwork arn(network);
	arn.set_learning_rate(learning_rate);
	return arn;
}

int	main(int argc, char **argv)
{
	try
	{
		int epoch = 1000;
		int batch = 1;
		std::string layer_function = "sigmoid";
		std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> data;
		ARNetwork arn = parse_args(argc, argv, data, layer_function, epoch, batch);
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}