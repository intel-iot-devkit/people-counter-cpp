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

#include "mqtt.h"

bool mqtt_initialized = false;
MQTTClient client;
MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
MQTTClient_message pubmsg = MQTTClient_message_initializer;
MQTTClient_deliveryToken token;
MQTTClient_SSLOptions sslOptions = MQTTClient_SSLOptions_initializer;

std::string std_getenv(const std::string &name)
{
    auto value = getenv(name.c_str());
    return value != nullptr ? std::string(value) : std::string();
}

void mqtt_init(mqtt_service_config const &config)
{
    if (mqtt_initialized)
    {
        return;
    }

    std::vector<char> server_c(
        config.server.c_str(),
        config.server.c_str() + config.server.size() + 1
    );

    std::vector<char> client_id_c(
        config.client_id.c_str(),
        config.client_id.c_str() + config.client_id.size() + 1
    );

    MQTTClient_create(&client,
                      &server_c[0],
                      &client_id_c[0],
                      MQTTCLIENT_PERSISTENCE_NONE,
                      NULL);

    // connection options
    conn_opts.keepAliveInterval = 20;
    conn_opts.cleansession = 1;

    if (!config.username.empty())
    {
        std::vector<char> username_c(
            config.username.c_str(),
            config.username.c_str() + config.username.size() + 1
        );
        
        conn_opts.username = &username_c[0];
    }

    if (!config.password.empty())
    {
        std::vector<char> password_c(
            config.password.c_str(),
            config.password.c_str() + config.password.size() + 1
        );
        
        conn_opts.password = &password_c[0];
    }

    // ssl options
    if (!config.cert.empty() && !config.cert_key.empty() && !config.ca_root.empty())
    {
        std::vector<char> cert_c(
            config.cert.c_str(),
            config.cert.c_str() + config.cert.size() + 1
        );

        std::vector<char> cert_key_c(
            config.cert_key.c_str(),
            config.cert_key.c_str() + config.cert_key.size() + 1
        );

        std::vector<char> ca_root_c(
            config.ca_root.c_str(),
            config.ca_root.c_str() + config.ca_root.size() + 1
        );
        
        sslOptions.keyStore = &cert_c[0];
        sslOptions.privateKey = &cert_key_c[0];
        sslOptions.trustStore = &ca_root_c[0];
    }
    else
    {
        sslOptions.enableServerCertAuth = false;
    };
    conn_opts.ssl = &sslOptions;

    mqtt_initialized = true;
};

int mqtt_start(MQTTClient_messageArrived* msgrcv)
{
    auto mqtt_config_result = get_mqtt_config();
    
    mqtt_service_config mqtt_config;
    bool mqtt_config_valid;
    
    std::tie(mqtt_config, mqtt_config_valid) = mqtt_config_result;
    
    if (!mqtt_config_valid)
    {
        return 1;
    }

    mqtt_init(mqtt_config);
    MQTTClient_setCallbacks(client, NULL, NULL, msgrcv, NULL);
    return 0;
}

void mqtt_close()
{
    if (mqtt_initialized)
    {
        //std::cout << "Closing MQTT..." << std::endl;
        MQTTClient_destroy(&client);
    }
};

void mqtt_connect()
{
    if (mqtt_initialized)
    {
        int rc;
        if ((rc = MQTTClient_connect(client, &conn_opts)) != MQTTCLIENT_SUCCESS)
        {
            //std::cout << "Failed to connect to MQTT server, return code:" << rc << std::endl;
            return;
        }
    }
}

void mqtt_disconnect()
{
    if (mqtt_initialized)
    {
        MQTTClient_disconnect(client, 10000);
    }
}

int mqtt_publish(std::string const &topic, std::string const &message)
{
    if (!mqtt_initialized) {
        return -1;
    }

    std::vector<char> topic_c(
        topic.c_str(),
        topic.c_str() + topic.size() + 1
    );

    std::vector<char> message_c(
        message.c_str(),
        message.c_str() + message.size() + 1
    );

    pubmsg.payload = &message_c[0];
    pubmsg.payloadlen = strlen(&message_c[0]);
    pubmsg.qos = QOS;
    pubmsg.retained = 0;
    int result = MQTTClient_publishMessage(client, &topic_c[0], &pubmsg, &token);
    if (result != 0) { // MQTTCLIENT_SUCCESS = 0
        return result;
    }
    return MQTTClient_waitForCompletion(client, token, TIMEOUT);
}

void mqtt_subscribe(std::string const &topic)
{
    if (!mqtt_initialized) {
        return;
    }

    MQTTClient_subscribe(client, topic.c_str(), 1);
}

std::pair<mqtt_service_config, bool> get_mqtt_config()
{
    const auto mqtt_server = std_getenv("MQTT_SERVER");
    const auto mqtt_client_id = std_getenv("MQTT_CLIENT_ID");
    const auto mqtt_username = std_getenv("MQTT_USERNAME");
    const auto mqtt_password = std_getenv("MQTT_PASSWORD");
    const auto mqtt_cert = std_getenv("MQTT_CERT");
    const auto mqtt_cert_key = std_getenv("MQTT_CERT_KEY");
    const auto mqtt_ca_root = std_getenv("MQTT_CA_ROOT");

    auto config_valid = false;

    mqtt_service_config config = {
        mqtt_server,
        mqtt_client_id,
        mqtt_username,
        mqtt_password,
        mqtt_cert,
        mqtt_cert_key,
        mqtt_ca_root};

    if (!mqtt_server.empty() && !mqtt_client_id.empty())
    {
        config_valid = true;
    }

    std::pair<mqtt_service_config, bool> result = {
        config,
        config_valid
    };

    return result;
}
