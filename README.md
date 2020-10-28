# People Counter

  | Details            |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 18.04 LTS   |
| Programming Language: |  C++ |
| Time to Complete:    |  45 min     |

![People Counter](./docs/images/people-counter-image.png)

## What It Does

This application is one of a series of IoT reference implementations illustrating how to develop a working solution for a particular problem. It demonstrates how to create a smart video IoT solution using Intel® hardware and software tools. This people counter solution detects people in a designated area, providing number of people in the frame, their average duration in the frame, and the total count.

## How It Works

The counter uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit. A trained neural network detects people within a designated area by displaying a bounding box over them. It counts the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame), and the total number of people detected, and then sends the data to a local web server using the Paho\* MQTT C client libraries.

![Architecture_diagram](./docs/images/arch_diagram.png)


## Requirements
### Hardware 
* 6th to 8th generation Intel® Core™ processors with Iris® Pro graphics or Intel® HD Graphics.

### Software
* [Ubuntu\* 18.04 LTS](http://releases.ubuntu.com/18.04/)<br><br>
**Note**: We recommend using a 4.14+ Linux* kernel with this software. Run the following command to determine your kernel version:

    ```
    uname -a
    ```
* OpenCL™ Runtime Package
* Intel® Distribution of OpenVINO™ toolkit 2020 R3 release
* Node v6.17.1
* Npm v3.10.10
* MQTT Mosca\* server

### Install Intel® Distribution of OpenVINO™ toolkit
Refer to [Install Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) to learn how to set up the toolkit.

Install the OpenCL™ Runtime Package to run inference on the GPU. It is not mandatory for CPU inference.

## Setup

### Get the code

Steps to clone the reference implementation:

```
sudo apt-get update && sudo apt-get install git
git clone https://github.com/intel-iot-devkit/people-counter-cpp.git 
```

### Which model to use

This application uses the [person-detection-retail-0013](https://docs.openvinotoolkit.org/2020.3/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) Intel® model, that can be accessed using the **model downloader**. The **model downloader** downloads the __.xml__ and __.bin__ files that will be used by the application.

### Install the dependencies

To install the dependencies of the RI and to download the **person-detection-retail-0013** Intel® model, run the following command:
```
cd <path_to_the_people-counter-cpp_directory>
./setup.sh
```

Make sure the npm and node versions are exact, using the commands given below:
```
node -v
```
The version should be **v6.17.1**

```
npm -v
```
The version should be **v3.10.10**

**Note**: If the Node and Npm versions are different, run the following commands:
```
sudo npm install -g n
sudo n 6.17.1
```

Note: After running the above commands, please open a new terminal to proceed further. Also, verify the node and npm versions from the new terminal.

## Installation
Go to people-counter-cpp directory:

```
cd <path_to_people-counter-cpp_directory>
```
### MQTT Mosca* server
   ```
   cd webservice/server
   npm install
   npm i jsonschema@1.2.6
   ```

### Web Server
  ```
  cd ../ui
  npm install
  ```

## Run the Application

There are three components that need to be running in separate terminals for this application to work:

-   MQTT Mosca server
-   Node.js* Web server
-   FFmpeg server

Go to people-counter-cpp directory:

```
cd <path_to_people-counter-cpp_directory>
```

### Step 1 - Start the Mosca server

Ensure that no process is running at port address 3000 using the following command:
```
sudo lsof -i:3000
```

Navigate to the `node-server` path and run the server using following commands:
```
cd webservice/server/node-server
node ./server.js
```
If successful, this message will appear in the terminal: 

```
connected to ./db/data.db
Mosca server started.
```

### Step 2 - Start the GUI

Open a new terminal and run the commands below:
```
cd ../../ui
npm run dev
```

If successful, this message will appear in the terminal:

```
webpack: Compiled successfully.
```

### Step 3 - FFmpeg Server

Open a new terminal and run the below commands:
```
cd ../..
sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Set Up the Environment
Open a new terminal in the current directory and run the below command to set up the environment variables required to run the Intel® Distribution of OpenVINO™ toolkit applications:
```
source /opt/intel/openvino/bin/setupvars.sh
```
**Note:** This command only needs to be executed once in the terminal where the application will be executed. If the terminal is closed, the command needs to be executed again. 

### Step 5 - Build and Start the Main Application 
This application uses the SSD derived person detection model bundled with the Intel® Distribution of OpenVINO™ toolkit.
To do a clean re-build, run the following commands:

```
cd ieservice
mkdir -p build && cd build
cmake ..
make
```

The new version of the software will be built as people-counter/ieservice/bin/intel64/Release/obj_recognition. Switch to the directory where the main application was built:

```
cd ../bin/intel64/Release
```

Set up the needed MQTT environment variables:

```
export MQTT_SERVER=localhost:1884
export MQTT_CLIENT_ID=cvservice
```

### Run on the CPU

```
./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -d CPU -thresh 0.65 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -i - http://localhost:8090/fac.ffm
```

**Note**:To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser. Refresh the browser window if the video does not play automatically.

### Run on the GPU

* To use GPU in 16-bit mode, use the following command:

    ```
    ./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -d GPU -thresh 0.65 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -i - http://localhost:8090/fac.ffm
    ```
    To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.

* To use GPU in 32-bit mode, use the following command:

    ```
    ./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -d GPU -thresh 0.65 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -i - http://localhost:8090/fac.ffm
    ```
    To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.

**Note**: The Loading time for GPU is more, so it might take few seconds to display the output. If request busy error is observed, please restart the ffmpeg server and try again.


### Run on the Intel® Neural Compute Stick
```
./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -d MYRIAD -thresh 0.65 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -i - http://localhost:8090/fac.ffm
```
To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.<br>

__Note:__ The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the -m <path_to_model> command-line argument, must be of data type FP16.

<!--
#### Run on the FPGA:

Before running the application on the FPGA, program the AOCX (bitstream) file.
Use the setup_env.sh script from [fpga_support_files.tgz](http://registrationcenter-download.intel.com/akdlm/irc_nas/12954/fpga_support_files.tgz) to set the environment variables.<br>

```
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```

The bitstreams for HDDL-F can be found under the `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams` folder. To program the bitstream use the below command:
```
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_RMNet.aocx
```
For more information on programming the bitstreams, please refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux-FPGA#inpage-nav-11.

To run on the FPGA, use the `-d HETERO:FPGA,CPU` command-line argument:
```
./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -d HETERO:FPGA,CPU -thresh 0.65 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -i - http://localhost:8090/fac.ffm
```
-->
### Use a Camera Stream
Use the camera ID followed by ```-i ``` , where the ID is taken from the video device (the number X in /dev/videoX). On Ubuntu, to list all available video devices use the following command:<br>
```
ls /dev/video*
```

For example, if the output of the above command is **/dev/video0**, then camera ID would be: **0**<br><br>
Run the application:

```
./obj_recognition -i 0 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -d CPU -thresh 0.65 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -i - http://localhost:8090/fac.ffm
```

To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.

**Note**: Use the camera's resolution with `-video_size` to observe the output on the web based interface. 
