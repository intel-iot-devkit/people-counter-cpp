/*
 * Copyright (c) 2018 Intel Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <gflags/gflags.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <sstream>
#include <vector>
#include <list>
#include <time.h>
#include <limits>
#include <chrono>
#include <string>

#include "common.hpp"
#include <cpp/ie_cnn_net_reader.h>
#include <ext_list.hpp>
#include <ie_plugin_ptr.hpp>
#include <ie_plugin_config.hpp>
#include <ie_extension.h>
#include <inference_engine.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

// MQTT
#include "mqtt.h"

using namespace cv;

using namespace InferenceEngine::details;
using namespace InferenceEngine;

#define DEFAULT_PATH_P "./lib"

#ifndef OS_LIB_FOLDER
#define OS_LIB_FOLDER "/"
#endif

// ---------------------------------------
// Define application options using gflags
// ---------------------------------------

/// @brief message for help argument
static const char help_message[] = "Print a usage message";
/// @brief message for images argument
static const char image_message[] = "Required. Path to input video file";
/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Path to a plugin folder";
/// @brief message for model argument
static const char model_message[] = "Required. Path to IR .xml file.";
/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. (MKLDNNPlugin, clDNNPlugin) Force load specified plugin ";
/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Infer target device (CPU or GPU or MYRIAD)";
/// @brief message for performance counters
static const char performance_counter_message[] = "Enables per-layer performance report";
/// @brief message for performance counters
static const char threshold_message[] = "confidence threshold for bounding boxes 0-1";
/// @brief message for batch size
static const char batch_message[] = "Batch size";
/// @brief message for frames count
static const char frames_message[] = "Number of frames from stream to process";

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);
/// \brief Define parameter for set video file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);
/// \brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);
/// \brief Define parameter for set plugin name <br>
/// It is a required parameter
DEFINE_string(p, "", plugin_message);
/// \brief Define parameter for set path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, DEFAULT_PATH_P, plugin_path_message);
/// \brief device the target device to infer on <br>
DEFINE_string(d, "", target_device_message);
/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);
/// \brief Enable per-layer performance report
DEFINE_double(thresh, .4, threshold_message);
/// \brief Batch size
DEFINE_int32(batch, 1, batch_message);
/// \brief Frames count
DEFINE_int32(fr, -1, frames_message);

std::string lastTopic, lastMESSAGE;
const int msgThrottle = 4;
int msgCounter = 0;
volatile bool performRegistration = false;
Mat img;


double getTime() {
	timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + (((double)t.tv_nsec) / 1.0e9);
}

// Publish MQTT message with a JSON payload
void publishMQTTMessage(const std::string& topic, const std::string& message)
{
	// Don't send repeat messages
	if (lastTopic == topic && lastMESSAGE == message && msgCounter++ % msgThrottle) {
		return;
	}

	lastTopic = topic;
	lastMESSAGE = message;

	mqtt_publish(topic, message);
}

// Message handler for the MQTT subscription for the "commands/register" topic
int handleControlMessages(void *context, char *topicName, int topicLen, MQTTClient_message *message)
{
	std::string topic = topicName;

	if (topic == "commands/register") {
		performRegistration = true;
	}
	return 1;
}

// This function show a help message
static void showUsage()
{
	std::cerr << std::endl;
	std::cerr << "Options:" << std::endl;
	std::cerr << std::endl;
	std::cerr << "    -h           " << help_message << std::endl;
	std::cerr << "    -i <path>    " << image_message << std::endl;
	std::cerr << "    -fr <path>   " << frames_message << std::endl;
	std::cerr << "    -m <path>    " << model_message << std::endl;
	std::cerr << "    -d <device>  " << target_device_message << std::endl;
	std::cerr << "    -pc          " << performance_counter_message << std::endl;
	std::cerr << "    -thresh <val>" << threshold_message << std::endl;
	std::cerr << "    -b <val>     " << batch_message << std::endl;
}

float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1 / 2;
	float l2 = x2 - w2 / 2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1 / 2;
	float r2 = x2 + w2 / 2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

float boxIntersection(DetectedObject a, DetectedObject b)
{
	float w = overlap(a.xmin, (a.xmax - a.xmin), b.xmin, (b.xmax - b.xmin));
	float h = overlap(a.ymin, (a.ymax - a.ymin), b.ymin, (b.ymax - b.ymin));

	if (w < 0 || h < 0) {
		return 0;
	}

	float area = w * h;

	return area;
}

float boxUnion(DetectedObject a, DetectedObject b)
{
	float i = boxIntersection(a, b);
	float u = (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin) - i;

	return u;
}

float boxIoU(DetectedObject a, DetectedObject b)
{
	return boxIntersection(a, b) / boxUnion(a, b);
}

void doNMS(std::vector<DetectedObject>& objects, float thresh)
{
	for(int i = 0; i < objects.size(); ++i){
		int any = 0;
		any = any || (objects[i].objectType > 0);
		if(!any) {
			continue;
		}

		for(int j = i + 1; j < objects.size(); ++j) {
			if (boxIoU(objects[i], objects[j]) > thresh) {
				if (objects[i].prob < objects[j].prob) {
					objects[i].prob = 0;
				}
				else {
					objects[j].prob = 0;
				}
			}
		}
	}
}

// Output BGR24 raw format to console.
void outputFrame(Mat img) {
	Vec3b pixel;
	for(int j = 0;j < img.rows;j++){
		for(int i = 0;i < img.cols;i++){
			pixel = img.at<Vec3b>(j, i);
			printf("%c%c%c", pixel[0], pixel[1], pixel[2]);
		}
	}
	fflush(stdout);
}

/**
* This function prints performance counters
* @param perfomanceMap - Map of performance counters
*/
void printPerformanceCounters(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfomanceMap) {
 	long totalTime = 0;
	std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>::const_iterator it;

	for (it = perfomanceMap.begin(); it != perfomanceMap.end(); ++it) {
		std::cerr << std::setw(30) << std::left << it->first + ":";
		switch (it->second.status) {
		case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
			std::cerr << std::setw(15) << std::left << "EXECUTED";
			break;
		case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
			std::cerr << std::setw(15) << std::left << "NOT_RUN";
			break;
		case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
			std::cerr << std::setw(15) << std::left << "OPTIMIZED_OUT";
			break;
		}

		if (it->second.realTime_uSec > 0) {
			totalTime += it->second.realTime_uSec;
		}
	}
}

/**
 * The main function of the ieservice application
 * @param argc - The number of arguments
 * @param argv - Arguments
 * @return 0 if all good
 */
int main(int argc, char *argv[]) {

	int result = mqtt_start(handleControlMessages);
	if(result !=0)
	{
		std::cout<<"MQTT not connected"<<std::endl;
		return 1;
	}

	mqtt_connect();
	mqtt_subscribe("person");

	// -----------------------------
	// Parse command line parameters
	// -----------------------------
	gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);

	if (FLAGS_h) {
		showUsage();
		return 1;
	}

	bool noPluginAndBadDevice = FLAGS_p.empty() && FLAGS_d.compare("CPU")
		&& FLAGS_d.compare("GPU") && FLAGS_d.compare("MYRIAD")
		&& FLAGS_d.compare("HETERO:FPGA,CPU") && FLAGS_d.compare("HETERO:HDDL,CPU");
	if (FLAGS_i.empty() || FLAGS_m.empty() || noPluginAndBadDevice) {
		if (noPluginAndBadDevice)
			std::cerr << "ERROR: device is not supported" << std::endl;
		if (FLAGS_m.empty())
			std::cerr << "ERROR: file with model - not set" << std::endl;
		if (FLAGS_i.empty())
			std::cerr << "ERROR: image(s) for inference - not set" << std::endl;
		showUsage();
		return 2;
	}

	//----------------------------------------------------------------------------
	// Prepare video input
	//----------------------------------------------------------------------------
	std::string input_filename = FLAGS_i;
	bool SINGLE_IMAGE_MODE = false;
	std::vector<std::string> imgExt = { ".bmp", ".jpg" };

	for (size_t i = 0; i < imgExt.size(); i++) {
		if (input_filename.rfind(imgExt[i]) != std::string::npos) {
			SINGLE_IMAGE_MODE = true;
			break;
		}
	}

	// Open video capture
	VideoCapture cap;
	if(strlen(FLAGS_i.c_str()) == 1){
		cap.open(std::stoi(FLAGS_i.c_str()));
		if (!cap.isOpened())   // Check if VideoCapture init successful
		{
			std::cerr << "Unable to open camera" << std::endl;
			return 1;
		}
	}
	else
	{
		cap.open(FLAGS_i.c_str());
		if (!cap.isOpened()){   // Check if VideoCapture init successful
			std::cerr << "Unable to open input file" << std::endl;
			return 1;
		}
	}

	// -----------------
	// Load plugin
	// -----------------
#ifdef WIN32
	std::string archPath = "../../../bin" OS_LIB_FOLDER "intel64/Release/";
#else
	std::string archPath = "../../../lib/" OS_LIB_FOLDER "intel64";
#endif

	InferenceEngine::InferenceEnginePluginPtr _plugin(selectPlugin(
		{
			FLAGS_pp,
			archPath,
			DEFAULT_PATH_P,
			""
			/* This means "search in default paths including LD_LIBRARY_PATH" */
		},
		FLAGS_p,
		FLAGS_d));

	const PluginVersion *pluginVersion;
	_plugin->GetVersion((const InferenceEngine::Version*&)pluginVersion);

	// ---------------------------
	// Enable performance counters
	// ---------------------------
	if (FLAGS_pc) {
		_plugin->SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } }, nullptr);
	}

	// ----------------
	// Read network
	// ----------------
	InferenceEngine::CNNNetReader network;
	try {
		network.ReadNetwork(FLAGS_m);
	}
	catch (InferenceEngineException ex) {
		std::cerr << "Failed to load network: " << ex.what() << std::endl;
		return 1;
	}

	std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
	network.ReadWeights(binFileName.c_str());

	// ---------------------
	// Set the target device
	// ---------------------
	if (!FLAGS_d.empty()) {
		network.getNetwork().setTargetDevice(getDeviceFromStr(FLAGS_d));
	}

	// --------------------
	// Set batch size
	// --------------------
	if (SINGLE_IMAGE_MODE) {
		network.getNetwork().setBatchSize(1);
	}
	else {
		network.getNetwork().setBatchSize(FLAGS_batch);
	}

	size_t batchSize = network.getNetwork().getBatchSize();

	//----------------------------------------------------------------------------
	//  Inference engine input setup
	//----------------------------------------------------------------------------

	// -----------------------
	// Set input configuration
	// -----------------------
	InputsDataMap inputs;
	inputs = network.getNetwork().getInputsInfo();

	if (inputs.size() != 1) {
		std::cerr << "This sample accepts networks having only one input." << std::endl;
		return 1;
	}

	InputInfo::Ptr ii = inputs.begin()->second;
	InferenceEngine::SizeVector inputDims = ii->getDims();

	if (inputDims.size() != 4) {
		std::cerr << "Not supported input dimensions size, expected 4, got "<< inputDims.size() << std::endl;
	}

	std::string imageInputName = inputs.begin()->first;

	DataPtr image = inputs[imageInputName]->getInputData();
	inputs[imageInputName]->setInputPrecision(Precision::FP32);

	// --------------------
	// Allocate input blobs
	// --------------------
	InferenceEngine::BlobMap inputBlobs;
	InferenceEngine::TBlob<float>::Ptr input =
	InferenceEngine::make_shared_blob < float,const InferenceEngine::SizeVector >(Precision::FP32, inputDims);
	input->allocate();

	inputBlobs[imageInputName] = input;

	int frame_width = inputDims[0];
	int frame_height = inputDims[1];

	// Use this to write the video to a file instead of stdout
	// VideoWriter video("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(frame_width, frame_height), true);

	// --------------------------------------------------------------------------
	// Load model into plugin
	// --------------------------------------------------------------------------

	// Add CPU Extensions
	if (FLAGS_d.compare("CPU") == 0) {
	// Required for support of certain layers in CPU
		InferencePlugin plugin(_plugin);
		plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
	}

	InferenceEngine::ResponseDesc dsc;
	InferenceEngine::StatusCode sts = _plugin->LoadNetwork(network.getNetwork(), &dsc);
	if (sts != 0) {
		std::cerr << "Error loading model into plugin: " << dsc.msg << std::endl;
		return 1;
	}

	//----------------------------------------------------------------------------
	//  Inference engine output setup
	//----------------------------------------------------------------------------

	// ---------------------
	// Get output dimensions
	// ---------------------
	InferenceEngine::OutputsDataMap out;
	out = network.getNetwork().getOutputsInfo();
	InferenceEngine::BlobMap outputBlobs;

	std::string outputName = out.begin()->first;
	int maxProposalCount = -1;

	for (auto && item : out) {
		InferenceEngine::SizeVector outputDims = item.second->dims;
		InferenceEngine::TBlob < float >::Ptr output;
		output = InferenceEngine::make_shared_blob < float,const InferenceEngine::SizeVector >(Precision::FP32,
				 outputDims);
		output->allocate();

		outputBlobs[item.first] = output;
		maxProposalCount = outputDims[1];
	}

	InferenceEngine::SizeVector outputDims = outputBlobs.cbegin()->second->dims();
	size_t outputSize = outputBlobs.cbegin()->second->size() / batchSize;

	//---------------------------
	// Main loop starts here
	//---------------------------
	Mat frame;
	int pplCount = 0;

	int currentCount = 0;
	int lastCount = 0;
	int totalCount = 0;
	double duration;
	double elapsedtime = getTime();

	Mat* resized = new Mat[batchSize];
	float normalize_factor = 1.0;

	auto input_channels = inputDims[2];  // Channels for color format RGB = 4
	size_t input_width = inputDims[1];
	size_t input_height = inputDims[0];
	auto channel_size = input_width * input_height;
	auto input_size = channel_size * input_channels;
	int totalFrames = 0;
	
	for (;;) {
		for (size_t mb = 0; mb < batchSize; mb++) {
			float* inputPtr = input.get()->data() + input_size * mb;

			//---------------------------
			// Get a new frame
			//---------------------------
			cap >> frame;
			totalFrames++;
			if (!frame.data) {
				exit(0);
			}

			//---------------------------------------------
			// Resize to expected size (in model .xml file)
			//---------------------------------------------
			resize(frame, resized[mb], Size(frame_width, frame_height));

			//-------------------------------------------------------
			// PREPROCESS STAGE:
			// Convert image to format expected by inference engine
			// IE expects planar, convert from packed
			//-------------------------------------------------------
			long unsigned int framesize = resized[mb].rows * resized[mb].step1();

			if (framesize != input_size) {
				std::cerr << "input pixels mismatch, expecting " << input_size << " bytes, got: " << framesize;
				return 1;
			}

			// imgIdx - Image pixel counter
			// channel_size - Size of a channel, computed as image size in bytes divided by number of channels, or image width * image height
			// inputPtr - A pointer to pre-allocated inout buffer
			for (size_t i = 0, imgIdx = 0, idx = 0; i < channel_size; i++, idx++) {
				for (size_t ch = 0; ch < input_channels; ch++, imgIdx++) {
					inputPtr[idx + ch * channel_size] = resized[mb].data[imgIdx] / normalize_factor;
				}
			}
		}

		if (FLAGS_fr > 0 && totalFrames > FLAGS_fr) {
			break;
		}

		//---------------------------
		// INFER STAGE
		//---------------------------
		sts = _plugin->Infer(inputBlobs, outputBlobs, &dsc);
		if (sts != 0) {
			std::cerr << "An infer error occurred: " << dsc.msg << std::endl;
			return 1;
		}

		//---------------------------
		// Read perfomance counters
		//---------------------------
		if (FLAGS_pc) {
			std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfomanceMap;
			_plugin->GetPerformanceCounts(perfomanceMap, nullptr);
			printPerformanceCounters(perfomanceMap);
		}

		//-----------------------------------------------
		// POSTPROCESS STAGE:
		// Parse output
		// Output layout depends on network topology
		//-----------------------------------------------
		InferenceEngine::Blob::Ptr detectionOutBlob = outputBlobs[outputName];
		const InferenceEngine::TBlob < float >::Ptr detectionOutArray =
			std::dynamic_pointer_cast <InferenceEngine::TBlob <
			float >>(detectionOutBlob);

		for (size_t mb = 0; mb < batchSize; mb++) {
			float *box = detectionOutArray->data() + outputSize * mb;
			std::vector < DetectedObject > detectedObjects;

				currentCount = 0;

				//---------------------------
				// Parse SSD output
				//---------------------------
				for (int c = 0; c < maxProposalCount; c++) {
					float image_id = box[c * 7 + 0];
					float label = box[c * 7 + 1];
					float confidence = box[c * 7 + 2];
					float xmin = box[c * 7 + 3] * inputDims[0];
					float ymin = box[c * 7 + 4] * inputDims[1];
					float xmax = box[c * 7 + 5] * inputDims[0];
					float ymax = box[c * 7 + 6] * inputDims[1];

					if (image_id < 0 || confidence == 0) {
						continue;
					}

					if (confidence > FLAGS_thresh) {

						if (label == 1.0){
							currentCount++;

							rectangle(resized[mb],
								Point((int)xmin, (int)ymin),
								Point((int)xmax, (int)ymax), Scalar(0, 55, 255),
								+1, 4);
						}
					}
				}

				if (currentCount > lastCount) {
					elapsedtime = getTime();

					totalCount += currentCount - lastCount;

					std::ostringstream payload;
					payload << "{\"total\":" << (int)totalCount << "}";
					std::string s = payload.str();
					publishMQTTMessage("person", s);
				}

				if (currentCount < lastCount) {
					duration = getTime() - elapsedtime;

					std::ostringstream payload;
					payload << "{\"duration\":" << (int)duration << "}";
					std::string s = payload.str();
					publishMQTTMessage("person/duration", s);
				}

				// Send the people count to MQTT server
				std::ostringstream payload;
				payload << "{\"count\":" << (int)currentCount << "}";
				std::string s = payload.str();
				publishMQTTMessage("person", s);

				lastCount = currentCount;
			} 

		//---------------------------
		// Display the output
		//---------------------------
		for (int mb = 0; mb < batchSize; mb++) {
			if (SINGLE_IMAGE_MODE) {
				imwrite("out.jpg", resized[mb]);
			}
			else {
				outputFrame(resized[mb]);
			}
		}
	}

	delete [] resized;
	return 0;
}
