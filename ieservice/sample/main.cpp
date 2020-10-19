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
#include <ie_plugin_ptr.hpp>	
#include <ie_plugin_config.hpp>
#include <inference_engine.hpp>
#include <ie_blob.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

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

bool isAsyncMode = true;

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
/// @brief message for async or sync mode
static const char async_message[] = "execution on SYNC or ASYNC mode. Default option is ASYNC mode";

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
DEFINE_double(thresh, .65, threshold_message);
/// \brief Batch size
DEFINE_int32(batch, 1, batch_message);
/// \brief Frames count
DEFINE_int32(fr, -1, frames_message);
/// \brief Enable sync or async mode
DEFINE_bool(f, true, async_message);

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
	std::cerr << "    -f 	       " << async_message << std::endl;
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
int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	// -----------------
	// Load plugin
	// -----------------
#ifdef WIN32
	std::string archPath = "../../../bin" OS_LIB_FOLDER "intel64/Release/";
#else
	std::string archPath = "../../../lib/" OS_LIB_FOLDER "intel64";
#endif

	// ----------------
	// Read network --  Network is created
	// ----------------

    InferenceEngine::Core ie;
    auto network = ie.ReadNetwork(FLAGS_m);

	// --------------------
	// Set batch size -- no need for setbatchsize
	// --------------------
        if (SINGLE_IMAGE_MODE) {
	network.setBatchSize(1);
	}
	else {
		network.setBatchSize(FLAGS_batch);
	}

	size_t batchSize = network.getBatchSize();

	//----------------------------------------------------------------------------
	//  Inference engine input setup
	//----------------------------------------------------------------------------

	// -----------------------
	// Set input configuration
	// -----------------------
	InputsDataMap inputInfo(network.getInputsInfo());

	InferenceEngine::SizeVector inputDims = inputInfo.begin()->second->getInputData()->getTensorDesc().getDims();

	std::string imageInputName, imageInfoInputName;
	size_t netInputHeight, netInputWidth, netInputChannel = 1;


	for (const auto & inputInfoItem : inputInfo)
	{
	if (inputInfoItem.second->getInputData()->getTensorDesc().getDims().size() == 4)
	{  // first input contains images
	    imageInputName = inputInfoItem.first;
	    inputInfoItem.second->setPrecision(Precision::U8);
	    inputInfoItem.second->getInputData()->setLayout(Layout::NCHW);
	    const TensorDesc& inputDesc = inputInfoItem.second->getTensorDesc();
	    netInputHeight = getTensorHeight(inputDesc);
	    netInputWidth = getTensorWidth(inputDesc);
	    netInputChannel = getTensorChannels(inputDesc);
	}
	else if (inputInfoItem.second->getTensorDesc().getDims().size() == 2)
	{  // second input contains image info
	    imageInfoInputName = inputInfoItem.first;
	    inputInfoItem.second->setPrecision(Precision::FP32);
	}
	else
	{
	    throw std::logic_error("Unsupported " +
		                   std::to_string(inputInfoItem.second->getTensorDesc().getDims().size()) + "D "
		                   "input layer '" + inputInfoItem.first + "'. "
		                   "Only 2D and 4D input layers are supported");
	}
	}

	OutputsDataMap outputInfo(network.getOutputsInfo());
	if (outputInfo.size() != 1) {
	throw std::logic_error("This demo accepts networks having only one output");
	}
	DataPtr& output = outputInfo.begin()->second;
	auto outputName = outputInfo.begin()->first;

	const SizeVector outputDims = output->getTensorDesc().getDims();
	const int maxProposalCount = outputDims[2];

	const int objectSize = outputDims[3];
	if (objectSize != 7) {
	throw std::logic_error("Output should have 7 as a last dimension");
	}
	if (outputDims.size() != 4) {
	throw std::logic_error("Incorrect output dimensions for SSD");
	}
	output->setPrecision(Precision::FP32);
	output->setLayout(Layout::NCHW);

	// -----------------------------------------------------------------------------------------------------

	// --------------------------- Loading model to the device ------------------------------------------

//	slog::info << "Loading model to the device" << slog::endl;
	ExecutableNetwork net = ie.LoadNetwork(network, FLAGS_d);

	// --------------------------- Create infer request -------------------------------------------------
	InferenceEngine::InferRequest::Ptr currInfReq = net.CreateInferRequestPtr();
	InferenceEngine::InferRequest::Ptr nextInfReq = net.CreateInferRequestPtr();

	//----------------------------------------------------------------------------

	if (!imageInfoInputName.empty())
	{
	auto setImgInfoBlob = [&](const InferRequest::Ptr &inferReq) {
	    auto blob = inferReq->GetBlob(imageInfoInputName);
	    auto data = blob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
	    data[0] = static_cast<float>(netInputHeight);  // height
	    data[1] = static_cast<float>(netInputWidth);  // width
	    data[2] = 1;
	};
	setImgInfoBlob(currInfReq);
	setImgInfoBlob(nextInfReq);
	}

	//---------------------------
	// Main loop starts here
	//---------------------------
	Mat frameInfer, prev_frame, frame, output_frames;
	int pplCount = 0;

	int currentCount = 0;
	int lastCount = 0;
	int totalCount = 0;
	double duration;
	double elapsedtime = getTime();

	float normalize_factor = 1.0;

	auto input_channels = netInputChannel;  // Channels for color format RGB = 4
	const size_t output_width = netInputWidth;
	const size_t output_height = netInputHeight;

	auto channel_size = output_width * output_height;
	auto input_size = channel_size * input_channels;
	int totalFrames = 0;
	
	for (;;) {

			//---------------------------
			// Get a new frame
			//---------------------------
			cap >> frame;
			totalFrames++;
			if (!frame.data) {
				exit(0);
			}

			Blob::Ptr inputBlob;
			//---------------------------------------------
			// Resize to expected size (in model .xml file)
			//---------------------------------------------
			resize(frame, output_frames, Size(netInputWidth, netInputHeight));
			frameInfer = output_frames;
			if(FLAGS_f)
			{
				inputBlob = nextInfReq->GetBlob(imageInputName);
				prev_frame = frame;
			}
			else
			{
				inputBlob = currInfReq->GetBlob(imageInputName);
				prev_frame = frame;
			}
			matU8ToBlob<uint8_t>(output_frames, inputBlob);

			//----------------------------------------------------
			// PREPROCESS STAGE:
			// convert image to format expected by inference engine
			// IE expects planar, convert from packed
			//----------------------------------------------------
			size_t framesize = frameInfer.rows * frameInfer.step1();

			if (framesize != input_size)
			{
				std::cout << "input pixels mismatch, expecting "
							<< input_size << " bytes, got: " << framesize
							<< std::endl;
				return 1;
			}



			//---------------------------
			// INFER STAGE
			//---------------------------

			if(FLAGS_f)
				nextInfReq->StartAsync();
			else
				currInfReq->StartAsync();

			if(OK == currInfReq->Wait(IInferRequest::WaitMode::RESULT_READY))
			{


				//---------------------------
				// Read perfomance counters
				//---------------------------
				if (FLAGS_pc) {
					std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfomanceMap;
					perfomanceMap = currInfReq->GetPerformanceCounts();
					printPerformanceCounters(perfomanceMap);
				}

				//---------------------------
				// POSTPROCESS STAGE:
				// Parse output
				//---------------------------
				float *box = currInfReq->GetBlob(outputName)->buffer().as<InferenceEngine::PrecisionTrait <InferenceEngine::Precision::FP32>::value_type *>();
                		LockedMemory<const void> outputMapped = as<MemoryBlob>(
                    		currInfReq->GetBlob(outputName))->rmap();
				currentCount = 0;
				//const float *box = outputMapped.as<float*>();

				//---------------------------
				// Parse SSD output
				//---------------------------
				for (int c = 0; c < maxProposalCount; c++) {
					float image_id = box[c * objectSize + 0];
					float label = static_cast<int>(box[c * objectSize + 1]);
					float confidence = box[c * objectSize + 2];
					float xmin = box[c * objectSize + 3] * width;
					float ymin = box[c * objectSize + 4] * height;
					float xmax = box[c * objectSize + 5] * width;
					float ymax = box[c * objectSize + 6] * height;

					if (image_id < 0 || confidence == 0) {
						continue;
					}

					if (confidence > FLAGS_thresh) {

						if (label == 1.0){
							currentCount++;
							rectangle(prev_frame,
								Point((int)xmin, (int)ymin),
								Point((int)xmax, (int)ymax), Scalar(0, 55, 255), +1, 4);
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
			if (SINGLE_IMAGE_MODE) {
				imwrite("out.jpg", prev_frame);
			}
			else {
				outputFrame(prev_frame);
			}

			if (FLAGS_f) {
				currInfReq.swap(nextInfReq);
				prev_frame = frame.clone();
			}

	}

	return 0;
}
