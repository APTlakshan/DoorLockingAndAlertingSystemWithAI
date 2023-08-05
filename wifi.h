#ifndef WIFI_H
#define WIFI_H

#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiServer.h>

extern const char* ssid;
extern const char* password;
extern String parameterValue;
extern WiFiServer server;

void wifi_setup();
void wifi_loop();
String getVal();
#endif  // WIFI_H;
