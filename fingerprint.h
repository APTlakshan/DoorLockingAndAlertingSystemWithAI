#ifndef FINGER_H
#define FINGER_H

#include <Adafruit_Fingerprint.h>
#include <HardwareSerial.h>

int getFingerprintIDez();
int fingerprint_loop();
int scan_finger();

extern int fprintx[3];
extern uint8_t count;

extern Adafruit_Fingerprint finger;
extern HardwareSerial mySerial;

void fingerprint_setup();

#endif  // FINGER_H
