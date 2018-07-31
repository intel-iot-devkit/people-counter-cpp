/*
* Copyright (c) 2015 - 2017 Intel Corporation.
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

#ifndef MQTT_H_INCLUDED
#define MQTT_H_INCLUDED

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <tuple>
#include <cstring>

extern "C" {
    #include "MQTTClient.h"
    #include "MQTTClientPersistence.h"
}

#define QOS 1
#define TIMEOUT 1000L

struct mqtt_service_config
{
    std::string server;
    std::string client_id;
    std::string topic;
    std::string username;
    std::string password;
    std::string cert;
    std::string cert_key;
    std::string ca_root;
};

std::string std_getenv(const std::string &name);
std::pair<mqtt_service_config, bool> get_mqtt_config();
int mqtt_start(MQTTClient_messageArrived* msgrcv);
void mqtt_close();
void mqtt_connect();
void mqtt_disconnect();
int mqtt_publish(std::string const &topic, std::string const &message);
void mqtt_subscribe(std::string const &topic);

#endif
