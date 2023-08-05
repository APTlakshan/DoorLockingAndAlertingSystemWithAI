#include "fingerprint.h"
#include "pins.h"
#include "wifi.h"

#include <Arduino.h>
#if (defined(AVR) || defined(ESP8266)) && !defined(AVR_ATmega2560)
HardwareSerial mySerial(1);
#else
#define mySerial Serial1
#endif

int fprintx[9];  // Array to store fingerprint IDs
uint8_t count = 0;  // Counter for the number of fingerprints stored
unsigned long startTime = 0;  // Variable to track the start time

Adafruit_Fingerprint finger(&mySerial);  // Create an instance of the Adafruit_Fingerprint class

void fingerprint_setup()
{
  Serial.begin(115200);  // Initialize the serial communication at 115200 baud rate
  while (!Serial);
  delay(100);
  Serial.println("\n\nAdafruit finger detect test");

  mySerial.begin(57600, SERIAL_8N1, 16, 17);  // Initialize the fingerprint sensor serial communication

  finger.begin(57600);  // Initialize the fingerprint sensor at 57600 baud rate
  delay(5);
  if (finger.verifyPassword()) {  // Verify the fingerprint sensor password
    Serial.println("Found fingerprint sensor!");
  } else {
    Serial.println("Did not find fingerprint sensor :(");
    while (1) {
      delay(1);
    }
  }

  finger.getParameters();  // Get fingerprint sensor parameters

  finger.getTemplateCount();  // Get the number of stored fingerprint templates
}

int fingerprint_loop()
{
  
  Serial.println("start");
  int var;
  uint8_t j;
  bool duplicate;

  while (count < 9) {  // Loop until 3 unique fingerprints are detected

    wifi_loop();
    duplicate = false;

    var = getFingerprintIDez();  // Get the fingerprint ID
    // Serial.println(".");

    j = 0;
    while (j < count) {  // Check if the fingerprint ID is already stored
    Serial.println("loop2");
      if (var == fprintx[j]) {
        duplicate = true;
        break;
      }
      j++;
    }

    if ((var != -1) && (duplicate == false)) {  // Store the unique fingerprint ID
      fprintx[count] = var;
      count++;
      digitalWrite(BUZZER_PIN, HIGH);  // Turn on the buzzer
      delay(200);
      digitalWrite(BUZZER_PIN, LOW);  // Turn off the buzzer
      delay(200);
    }
  }

  if (count == 9) {  // If 3 unique fingerprints are detected
    count = 0;
    delay(500);
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);

    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);
    return 1;  // Return success
  }

  delay(500);
  
  Serial.println("stop");
  return -1;  // Return failure
}

int scan_finger()
{
  fingerprint_setup();  // Initialize the fingerprint sensor
  startTime = millis();  // Get the current time

  while (millis() - startTime < 20000) {  // Perform fingerprint scanning for a maximum of 20 seconds
    int result = fingerprint_loop();  // Execute fingerprint scanning loop
    if (result == 1) {
      // Fingerprint scan successful
      return 1;
    }
  }
  // Timeout reached, no fingerprint detected
  return 2;
}

int getFingerprintIDez()
{
  uint8_t p = finger.getImage();  // Capture fingerprint image

  if (p != FINGERPRINT_OK) return -1;

  p = finger.image2Tz();  // Convert image to fingerprint template
  if (p != FINGERPRINT_OK) return -1;

  p = finger.fingerFastSearch();  // Search for the fingerprint in the database
  if (p != FINGERPRINT_OK) return -1;

  Serial.print("Found ID #");
  Serial.print(finger.fingerID);  // Print the matched fingerprint ID
  Serial.print(" with confidence of ");
  Serial.println(finger.confidence);  // Print the confidence level of the match

  return finger.fingerID;  // Return the matched fingerprint ID
}
