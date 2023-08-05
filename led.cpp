#include "led.h"
#include <Arduino.h>
void sucLED()
{
  digitalWrite(S_LED, HIGH);
  delay(1000);
  digitalWrite(S_LED, LOW);
  delay(1000);
}

void fLED()
{
  digitalWrite(F_LED, HIGH);
  delay(20000);
  digitalWrite(F_LED, LOW);
  delay(20000);
}
