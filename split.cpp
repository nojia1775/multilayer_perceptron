#include "ARNetwork/neural_network/include/ARNetwork.hpp"

static void	read_line(const std::string& line, std::vector<double>& data)
{
	bool comma = false;
	for (size_t i = 0 ; line[i] ; i++)
	{
		if (line[i] == ',')
			comma = true;
		if (isalpha(line[i]) && comma && line[i - 1] == ',')
			data.push_back(line[i] == 'M' ? 1 : 0);
		else if (comma && line[i - 1] == ',')
			data.push_back(std::atof(line.c_str() + i));
	}
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
			if ((!isdigit(line[i - 1]) && line[i - 1] != 'M' && line[i - 1] != 'B') || (!isdigit(line[i + 1]) && line[i + 1] != 'M' && line[i + 1] != 'B'))
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

static std::vector<std::vector<double>>	read_file(const std::string& dataset)
{
	std::ifstream file(dataset);
	if (!file)
		throw Error("Error: couldn't open " + dataset);
	std::vector<std::vector<double>> data;
	std::string line;
	size_t count_line = 0;
	while (getline(file, line))
	{
		valid_line(line, 31, 30, dataset, count_line++);
		std::vector<double> dataline;
		read_line(line, dataline);
		data.push_back(dataline);
	}
	file.close();
	return data;
}

static void	normalize_data(std::vector<std::vector<double>>& data)
{
	size_t nbr_line = data.size();
	size_t nbr_columns = data[0].size();
	std::vector<double> mean_coefs(nbr_columns, 0);
	std::vector<double> max_coef(nbr_columns, 0);
	for (const auto& line : data)
	{
		for (size_t col = 0 ; col < nbr_columns ; col++)
		{
			mean_coefs[col] += line[col];
			max_coef[col] = line[col] > max_coef[col] ? line[col] : max_coef[col];
		}
	}
	for (auto& coef : mean_coefs)
		coef /= nbr_line;
	for (auto& line : data)
	{
		for (size_t i = 0 ; i < nbr_columns ; i++)
		{
			if (i != 0)
			{
				line[i] = line[i] == 0 ? mean_coefs[i] : line[i];
				line[i] /= max_coef[i];
			}
		}
	}
}

static void	split(const int& training_percentage, const std::vector<std::vector<double>>& data)
{
	if (training_percentage < 0 || training_percentage > 100)
		throw Error("Error: training percentage must be between 0 and 100");
	std::ofstream training_file("training.csv");
	std::ofstream validation_file("validation.csv");
	if (!training_file || !validation_file)
		throw Error("Error: couldn't open training.csv or validation.csv");
	double train_lines = (double)data.size() * ((double)training_percentage / 100.0);
	size_t line = 0;
	std::vector<std::vector<double>> tmp = data;
	for ( ; line < (size_t)train_lines ; line++)
	{
		size_t random_line = (size_t)random_double(0, tmp.size());
		for (size_t coef = 0 ; coef < tmp[random_line].size() ; coef++)
		{
			training_file << tmp[random_line][coef];
			if (coef == tmp[random_line].size() - 1)
				training_file << std::endl;
			else
				training_file << ',';
		}
		std::vector<std::vector<double>>::const_iterator it = tmp.begin();
		size_t count = 0;
		for ( ; it != tmp.end() && count < random_line ; ++it)
			count++;
		tmp.erase(it);
	}
	for ( ; tmp.size() ; )
	{
		size_t random_line = (size_t)random_double(0, tmp.size());
		for (size_t coef = 0 ; coef < tmp[random_line].size() ; coef++)
		{
			validation_file << tmp[random_line][coef];
			if (coef == tmp[random_line].size() - 1)
				validation_file << std::endl;
			else
				validation_file << ',';
		}
		std::vector<std::vector<double>>::const_iterator it = tmp.begin();
		size_t count = 0;
		for ( ; it != tmp.end() && count < random_line ; ++it)
			count++;
		tmp.erase(it);
	}
	training_file.close();
	validation_file.close();
}

int	main(int argc, char **argv)
{
	try
	{
		if (argc != 2)
			throw Error("Error: ./split <training_percentage>");
		std::vector<std::vector<double>> data = read_file("data.csv");
		normalize_data(data);
		split(std::atoi(argv[1]), data);
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}