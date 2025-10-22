#include "ARNetwork/neural_network/include/ARNetwork.hpp"

int	main(int argc, char **argv)
{
	try
	{
		if (argc != 33)
			throw Error("Error: ./prediction <file.json> <layer_function> [30 datas]");
		Vector<double> inputs(30);
		for (size_t i = 0 ; i < 30 ; i++)
			inputs[i] = std::atof(argv[i + 3]);
		ARNetwork arn(argv[1]);
		Vector<double> outputs = arn.feed_forward(inputs, argv[2], "softmax");
		outputs.display();
	}
	catch (const std::exception& e) { std::cerr << e.what() << std::endl; }
	return 0;
}