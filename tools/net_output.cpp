#include <opencv2/opencv.hpp>
#include <cstdio>
#include <string>
#include <map>
#include <net.h>
#include <regex>
#include <algorithm>
#include <numeric>

#define BUF_SIZE 1024

struct PreParam
{
	float mean[3];
	float norm[3];
	int width;
	int height;
	bool swapRB;
};



static int find_all_value_in_string(const std::string& values_string, std::vector<float>& value)
{
	std::vector<int> masks_pos;

	for (size_t i = 0; i < values_string.size(); i++)
	{
		if (',' == values_string[i])
		{
			masks_pos.push_back(static_cast<int>(i));
		}
	}

	// check
	if (masks_pos.empty())
	{
		fprintf(stderr, "ERROR: Cannot find any ',' in string, please check.\n");
		return -1;
	}

	if (2 != masks_pos.size())
	{
		fprintf(stderr, "ERROR: Char ',' in fist of string, please check.\n");
		return -1;
	}

	if (masks_pos.front() == 0)
	{
		fprintf(stderr, "ERROR: Char ',' in fist of string, please check.\n");
		return -1;
	}

	if (masks_pos.back() == 0)
	{
		fprintf(stderr, "ERROR: Char ',' in last of string, please check.\n");
		return -1;
	}

	for (size_t i = 0; i < masks_pos.size(); i++)
	{
		if (i > 0)
		{
			if (!(masks_pos[i] - masks_pos[i - 1] > 1))
			{
				fprintf(stderr, "ERROR: Neighbouring char ',' was found.\n");
				return -1;
			}
		}
	}

	const cv::String ch0_val_str = values_string.substr(0, masks_pos[0]);
	const cv::String ch1_val_str = values_string.substr(masks_pos[0] + 1, masks_pos[1] - masks_pos[0] - 1);
	const cv::String ch2_val_str = values_string.substr(masks_pos[1] + 1, values_string.size() - masks_pos[1] - 1);

	value.push_back(static_cast<float>(std::atof(std::string(ch0_val_str).c_str())));
	value.push_back(static_cast<float>(std::atof(std::string(ch1_val_str).c_str())));
	value.push_back(static_cast<float>(std::atof(std::string(ch2_val_str).c_str())));

	return 0;
}


void showUsage()
{
	std::cout << "example1: ./ncnn2table --help" << std::endl;
	std::cout << "example2: ./ncnn2table --param=squeezenet-fp32.param --bin=squeezenet-fp32.bin --label=label.txt --annotation=annotation.txt" << std::endl;
}

void pretty_print(const ncnn::Mat& m)
{
	for (int q = 0; q < m.c; q++)
	{
		const float* ptr = m.channel(q);
		for (int y = 0; y < m.h; y++)
		{
			for (int x = 0; x < m.w; x++)
			{
				printf("%f ", ptr[x]);
			}
			ptr += m.w;
			printf("\n");
		}
		printf("------------------------\n");
	}
}

int main(int argc, char** argv)
{
	std::cout << __TIME__ << std::endl;

	const char* key_map =
		"{help h usage ? |    | print this message}"
		"{param p        |    | path to ncnn.param file}"
		"{bin b          |    | path to ncnn.bin file}"
		"{label l        |    | path to image label file}"
		"{annotation a   |    | path to image annotation file}"
		"{output o       |    | path to the result }"
		"{input_name i   |data| the network input name}"
		"{output_name    |prob| the network output name}"
		"{mean m         |    | value of mean (mean value, default is 104.0,117.0,123.0) }"
		"{norm n         |    | value of normalize (scale value, default is 1.0,1.0,1.0) }"
		"{size s         |    | the size of input image(using the resize the original image,default is w=224,h=224) }"
		"{swapRB c       |    | flag which indicates that swap first and last channels in 3-channel image is necessary }"
		"{thread t       |  4 | count of processing threads }";


	cv::CommandLineParser parser(argc, argv, key_map);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}

	//if (!parser.has("param") || !parser.has("bin") || !parser.has("label") || !parser.has("annotation"))
	//{
	//	std::cout << "Inputs is does not include all needed param, pleas check..." << std::endl;
	//	parser.printMessage();
	//	showUsage();
	//	return 0;
	//}

	const std::string label_file_path = parser.get<cv::String>("label");
	const std::string annotation_file_path = parser.get<cv::String>("annotation");
	const std::string ncnn_param_file_path = parser.get<cv::String>("param");
	const std::string ncnn_bin_file_path = parser.get<cv::String>("bin");
	const std::string saved_result_file_path = parser.get<cv::String>("output");
	const std::string ncnn_net_input_name = parser.get<cv::String>("input_name");
	const std::string ncnn_net_output_name = parser.get<cv::String>("output_name");


	// check the input param
	//if (label_file_path.empty() || annotation_file_path.empty() || ncnn_param_file_path.empty() || ncnn_bin_file_path.empty() || saved_result_file_path.empty())
	//{
	//	fprintf(stderr, "One or more path may be empty, please check and try again.\n");
	//	return 0;
	//}

	const int num_threads = parser.get<int>("thread");

	struct PreParam pre_param;
	pre_param.mean[0] = 0.f;
	pre_param.mean[1] = 0.f;
	pre_param.mean[2] = 0.f;
	pre_param.norm[0] = 1.f;
	pre_param.norm[1] = 1.f;
	pre_param.norm[2] = 1.f;
	pre_param.width = 224;
	pre_param.height = 224;
	pre_param.swapRB = false;

	if (parser.has("mean"))
	{
		const std::string mean_str = parser.get<std::string>("mean");

		std::vector<float> mean_values;
		const int ret = find_all_value_in_string(mean_str, mean_values);
		if (0 != ret && 3 != mean_values.size())
		{
			fprintf(stderr, "ERROR: Searching mean value from --mean was failed.\n");

			return -1;
		}

		pre_param.mean[0] = mean_values[0];
		pre_param.mean[1] = mean_values[1];
		pre_param.mean[2] = mean_values[2];
	}

	if (parser.has("norm"))
	{
		const std::string norm_str = parser.get<std::string>("norm");

		std::vector<float> norm_values;
		const int ret = find_all_value_in_string(norm_str, norm_values);
		if (0 != ret && 3 != norm_values.size())
		{
			fprintf(stderr, "ERROR: Searching mean value from --mean was failed, please check --mean param.\n");

			return -1;
		}

		pre_param.norm[0] = norm_values[0];
		pre_param.norm[1] = norm_values[1];
		pre_param.norm[2] = norm_values[2];
	}

	if (parser.has("size"))
	{
		cv::String size_str = parser.get<std::string>("size");

		size_t sep_pos = size_str.find_first_of(',');

		if (cv::String::npos != sep_pos && sep_pos < size_str.size())
		{
			cv::String width_value_str;
			cv::String height_value_str;

			width_value_str = size_str.substr(0, sep_pos);
			height_value_str = size_str.substr(sep_pos + 1, size_str.size() - sep_pos - 1);

			pre_param.width = static_cast<int>(std::atoi(std::string(width_value_str).c_str()));
			pre_param.height = static_cast<int>(std::atoi(std::string(height_value_str).c_str()));
		}
		else
		{
			fprintf(stderr, "ERROR: Searching size value from --size was failed, please check --size param.\n");

			return -1;
		}
	}

	if (parser.has("swapRB"))
	{
		pre_param.swapRB = true;
	}


	ncnn::Option default_option;
	default_option.use_int8_inference = true;
	default_option.use_int8_arithmetic = true;

	ncnn::Net net;
	net.load_param(ncnn_param_file_path.c_str());
	net.load_model(ncnn_bin_file_path.c_str());
	net.opt = default_option;
	ncnn::Mat in(pre_param.width, pre_param.height, 3),feat;
	in.fill(0.01f);


	ncnn::Extractor ex = net.create_extractor();
	ex.input(ncnn_net_input_name.c_str(), in);
	ex.extract(ncnn_net_output_name.c_str(), feat);

	pretty_print(feat);


	return 0;
}