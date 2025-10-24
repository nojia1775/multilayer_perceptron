#include "ARNetwork/neural_network/include/ARNetwork.hpp"

static std::vector<size_t>	get_network(const std::string& arg)
{
	std::vector<size_t> layers;
	layers.push_back(30);
	for (size_t i = 0 ; i < arg.size() ; i++)
	{
		if (!isdigit(arg[i]) && arg[i] != ' ')
			throw Error("Error: wrong format of layer");
		if (isdigit(arg[i]))
		{
			layers.push_back(std::atoi(arg.c_str() + i));
			for ( ; i < arg.size() ; i++)
				if (arg[i] == ' ')
					break;
		}
	}
	layers.push_back(2);
	return layers;
}

static ARNetwork	parse_args(int argc, char **argv, std::string& layer_function, int& epoch, int& batch)
{
	if (argc == 1)
		throw Error("Error: ./train --layer '<layers>' [--epoch <epoch> --learning_rate <learning_rate> --layer_function <layer_function> --batch <batch>]");
	double learning_rate = 0.1;
	std::vector<size_t> network;
	for (size_t i = 1 ; (int)i < argc && argv[i] ; i += 2)
	{
		if (std::string(argv[i]) == "--epoch")
		{
			if (!argv[i + 1])
				throw Error("Error: ./train --layer '<layers>' [--epoch <epoch> --learning_rate <learning_rate> --layer_function <layer_function> --batch <batch>]");
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
				throw Error("Error: ./train --layer '<layers>' [--epoch <epoch> --learning_rate <learning_rate> --layer_function <layer_function> --batch <batch>]");
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
				throw Error("Error: ./train --layer '<layers>' [--epoch <epoch> --learning_rate <learning_rate> --layer_function <layer_function> --batch <batch>]");
			layer_function = argv[i + 1];
		}
		else if (std::string(argv[i]) == "--batch")
		{
			if (!argv[i + 1])
				throw Error("Error: ./train --layer '<layers>' [--epoch <epoch> --learning_rate <learning_rate> --layer_function <layer_function> --batch <batch>]");
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
				throw Error("Error: ./train --layer '<layers>' [--epoch <epoch> --learning_rate <learning_rate> --layer_function <layer_function> --batch <batch>]");
			network = get_network(argv[i + 1]);
		}
		else
			throw Error("Error: unknown flag: " + std::string(argv[i]));
		if (network.empty())
			throw Error("Error: layers are missing");
	}
	ARNetwork arn(network);
	arn.set_learning_rate(learning_rate);
	return arn;
}

static void	valid_line(const std::string& line, const size_t& comma, const size_t& dot, const std::string& file, const size_t& index)
{
	size_t count_dot = 0;
	size_t count_comma = 0;
	for (size_t i = 0 ; i < line.size() ; i++)
	{
		if (!isdigit(line[i]) && line[i] != ',' && line[i] != '.' && line[i] != 'M' && line[i] != 'B')
			throw Error("Error: " + file + std::string(" is corrupted: line " + index) + std::string(" column " + i));
		if (line[i] == ',')
		{
			count_comma++;
			if (i == 0)
				throw Error("Error: " + file + std::string(" is corrupted: line " + index) + std::string(" column " + i));
			if ((!isdigit(line[i - 1]) && line[i - 1] != 'M' && line[i - 1] != 'B') || (!isdigit(line[i + 1]) && line[i - 1] != 'M' && line[i - 1] != 'B'))
				throw Error("Error: " + file + std::string(" is corrupted: line " + index) + std::string(" column " + i));
		}
		if (line[i] == '.')
		{
			count_dot++;
			if (i == 0)
				throw Error("Error: " + file + std::string(" is corrupted: line " + index) + std::string(" column " + i));
			if (!isdigit(line[i - 1]) || !isdigit(line[i + 1]))
				throw Error("Error: " + file + std::string(" is corrupted: line " + index) + std::string(" column " + i));
		}
	}
	if (count_comma != comma || count_dot > dot)
		throw Error("Error: " + file + std::string(" is corrupted: wrong number of comma or dot"));
}

static std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>	extract_datas(const std::string& csv)
{
	std::ifstream file(csv);
	if (!file)
		throw Error("Error: couldn't open " + csv);
	std::string line;
	std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> datas;
	size_t count_line = 0;
	while (getline(file, line))
	{
		std::vector<double> output;
		std::vector<double> input;
		valid_line(line, 30, 30, csv, count_line++);
		for (size_t i = 0 ; i < line.size() ; i++)
		{
			int malin;
			if (i == 0)
			{
				malin = std::atof(line.c_str());
				if (malin != 1 && malin != 0)
					throw Error("Error: " + csv + std::string(" is corrupted"));
				if (malin)
					output = {0.0, 1.0};
				else
					output = {1.0, 0.0};
			}
			if (line[i - 1] == ',')
				input.push_back(std::atof(line.c_str() + i));
		}
		datas.first.push_back(input);
		datas.second.push_back(output);
	}
	return datas;
}

int	main(int argc, char **argv)
{
	try
	{
		int epoch = 1000;
		int batch = 1;
		std::string layer_function = "sigmoid";
		ARNetwork arn = parse_args(argc, argv, layer_function, epoch, batch);
		arn.randomize_bias(0, -sqrt(6 / 43), sqrt(6 / 43));
		arn.randomize_weights(0, -sqrt(6 / 43), sqrt(6 / 43));
		arn.randomize_bias(1, -sqrt(6 / 24), sqrt(6 / 24));
		arn.randomize_weights(1, -sqrt(6 / 24), sqrt(6 / 24));
		arn.randomize_bias(2, -sqrt(6 / 10), sqrt(6 / 10));
		arn.randomize_weights(2, -sqrt(6 / 10), sqrt(6 / 10));
		std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> train_datas = extract_datas("training.csv");
		std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> validation_datas = extract_datas("validation.csv");
		std::pair<std::map<size_t, std::pair<double, double>>, std::map<size_t, std::pair<double, double>>> tracking = arn.train("bce", layer_function, "softmax", {ARNetwork::batching(train_datas.first, batch), ARNetwork::batching(validation_datas.first, batch)}, {ARNetwork::batching(train_datas.second, batch), ARNetwork::batching(validation_datas.second, batch)}, epoch);
		arn.get_json("model.json");
		for (const auto& track : tracking.first)
			std::cout << track.first << " loss = " << track.second.first << " r2 = " << track.second.second << std::endl;
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}