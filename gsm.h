#ifndef GSM_H
#define GSM_H

#include <HardwareSerial.h>
#include "pins.h"

extern HardwareSerial mySerial;

void call();
void message();
void gsm_setup();
void gsm_loop();

#endif  // GSM_H
