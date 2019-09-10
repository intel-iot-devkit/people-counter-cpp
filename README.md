
# People Counter

  | Details            |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 16.04 LTS   |
| Programming Language: |  C++ |
| Time to Complete:    |  45 min     |

![People Counter](./docs/images/people-counter-image.png)

This reference implementation is also [available in Python](https://github.com/intel-iot-devkit/people-counter-python)

## What It Does

This people counter application is one of a series of IoT reference implementations illustrating how to develop a working solution for a particular problem. It demonstrates how to create a smart video IoT solution using Intel® hardware and software tools. This people counter solution detects people in a designated area, providing number of people in the frame, their average duration in the frame, and the total count.

## How It Works

The counter uses the Inference Engine included in the Intel® Distribution of OpenVINO™ toolkit. A trained neural network detects people within a designated area by displaying a bounding box over them. It counts the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame), and the total number of people detected, and then sends the data to a local web server using the Paho\* MQTT C client libraries.

![Architecture_diagram](./docs/images/arch_diagram.png)


## Requirements
### Hardware 
* 6th to 8th generation Intel® Core™ processors with Iris® Pro graphics or Intel® HD Graphics.

### Software
* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)<br><br>
**Note**: We recommend using a 4.14+ Linux* kernel with this software. Run the following command to determine your kernel version:

    ```
    uname -a
    ```
* OpenCL™ Runtime Package
* Intel® Distribution of OpenVINO™ toolkit 2019 R1 release
* Node v6.17.1
* Npm v3.10.10
* MQTT Mosca\* server


## Setup

In order to work, the application requires four components running in separate terminals:

* MQTT Mosca server
* Node.js Web server 
* FFmpeg server
* Computer vision application (ieservice/bin/intel64/Release/obj_recognition)

**Note**: Run each in a separate terminal, or using something like tmux.

Before running the MQTT or web server, install the following dependencies:
```
sudo apt update
sudo apt install libzmq3-dev libkrb5-dev
```

### Install Nodejs and its dependencies

- This step is only required if the user previously used Chris Lea's Node.js PPA.

	```
	sudo add-apt-repository -y -r ppa:chris-lea/node.js
	sudo rm -f /etc/apt/sources.list.d/chris-lea-node_js-*.list
	sudo rm -f /etc/apt/sources.list.d/chris-lea-node_js-*.list.save
	```
- To install Nodejs and Npm, run the below commands:
	```
	curl -sSL https://deb.nodesource.com/gpgkey/nodesource.gpg.key | sudo apt-key add -
	VERSION=node_6.x
	DISTRO="$(lsb_release -s -c)"
	echo "deb https://deb.nodesource.com/$VERSION $DISTRO main" | sudo tee /etc/apt/sources.list.d/nodesource.list
	echo "deb-src https://deb.nodesource.com/$VERSION $DISTRO main" | sudo tee -a /etc/apt/sources.list.d/nodesource.list
	sudo apt-get update
	sudo apt-get install nodejs
	```

### Install Intel® Distribution of OpenVINO™ toolkit
Refer to [Install Intel® Distribution of OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux) to learn how to set up the toolkit.

Install the OpenCL™ Runtime Package to run inference on the GPU. It is not mandatory for CPU inference.

### Install Paho\* MQTT C client libraries

```
sudo apt update
sudo apt install libssl-dev
sudo apt-get install doxygen graphviz
cd ~
git clone https://github.com/eclipse/paho.mqtt.c.git
cd paho.mqtt.c
make
make html
sudo make install
sudo ldconfig
```
## Download the model
This application uses the **person-detection-retail-0013** Intel® model, that can be downloaded using the model downloader. The model downloader downloads the .xml and .bin files that will be used by the application.

Steps to download .xml and .bin files:
* Go to the **model_downloader** directory using the following command:
    ```
    cd /opt/intel/openvino/deployment_tools/tools/model_downloader
    ```

* Specify which model to download with __--name__:
    ```
    sudo ./downloader.py --name person-detection-retail-0013
    ```
* To download the model for FP16, run the following command:
    ```
    sudo ./downloader.py --name person-detection-retail-0013-fp16
    ```

The files will be downloaded inside the `/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt` directory.

## Installation
Go to people-counter directory:

```
cd <path_to_people-counter_directory>
```
### MQTT Mosca* server
   ```
   cd webservice/server
   npm install
   ```

   If any configuration errors occur while using **npm install**, use the commands below:
   ```
   npm config set registry "http://registry.npmjs.org"
   npm install
   ```
### Web Server
  ```
  cd ../ui
  npm install
  ```

### FFmpeg Server
This reference implementation uses ffmpeg to compress and stream video output from cvservice to the webservice clients. FFmpeg is installed separately from the Ubuntu repositories:

```
sudo apt update
sudo apt install ffmpeg
```

## Run the Application

### Step 1 - Start the Mosca server
```
cd ../server/node-server
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
./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -d CPU -thresh 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 544x320 -i - http://localhost:8090/fac.ffm
```

To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.

### Run on the GPU

* To use GPU in 16-bit mode, use the following command:

    ```
    ./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -d GPU -thresh 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 544x320 -i - http://localhost:8090/fac.ffm
    ```
    To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.

* To use GPU in 32-bit mode, use the following command:

    ```
    ./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -d GPU -thresh 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 544x320 -i - http://localhost:8090/fac.ffm
    ```
    To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.


### Run on the Intel® Neural Compute Stick
```
./obj_recognition -i Pedestrain_Detect_2_1_1.mp4 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013-fp16.xml -d MYRIAD -thresh 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 544x320 -i - http://localhost:8090/fac.ffm
```
To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.<br>

__Note:__ The Intel® Neural Compute Stick can only run FP16 models. The model that is passed to the application, through the -m <path_to_model> command-line argument, must be of data type FP16.


### Use a Camera Stream
Use the camera ID followed by ```-i ``` , where the ID is taken from the video device (the number X in /dev/videoX). On Ubuntu, to list all available video devices use the following command:<br>
```
ls /dev/video*
```

For example, if the output of the above command is **/dev/video0**, then camera ID would be: **0**<br><br>
Run the application:

```
./obj_recognition -i 0 -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -d CPU -thresh 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 544x320 -i - http://localhost:8090/fac.ffm
```

To see the output on web based interface, open the link [http://localhost:8080](http://localhost:8080/) on browser.
