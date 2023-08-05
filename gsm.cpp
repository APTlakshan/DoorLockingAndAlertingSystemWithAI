#include "gsm.h"

HardwareSerial mySerial(0); // RX, TX

void call()
{
  mySerial.println("AT");
  delay(1000);
  mySerial.println("ATD+94761776047;"); // Replace with your phone number
}

void message()
{
  gsm_setup();
  gsm_loop();
  mySerial.println("AT+CMGF=1"); // Configuring TEXT mode
  delay(1000);
  mySerial.println("AT+CMGS=\"+94760140282\""); // Replace with recipient's phone number
  delay(1000);
  mySerial.print("Unauthorized Entry..!!"); // Text content
  delay(1000);
  mySerial.write(26);
}

void gsm_setup()
{
  Serial.begin(115200);
  mySerial.begin(57600, SERIAL_8N1, 3,1);
  while (!Serial) {
    ; // Wait for serial port to connect. Needed for native USB port only
  }

  Serial.println("Initializing...");
  mySerial.begin(115200);
  delay(1000);

  // call();
  // delay(15000);

  // message();
}

void gsm_loop()
{
  if (mySerial.available()) {
    Serial.write(mySerial.read());
  }
  if (Serial.available()) {
    mySerial.write(Serial.read());
  }
}
