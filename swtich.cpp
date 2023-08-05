#include "switch.h"
#include <Arduino.h>
#include "pins.h"
void lockFunc()
{
  digitalWrite(LOCK_PIN, HIGH);
   digitalWrite(S_LED, HIGH);
  //delay(1000);
  for (int i = 0; i < 20; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(250);
    digitalWrite(BUZZER_PIN, LOW);
    delay(250);
  }
  digitalWrite(LOCK_PIN, LOW);
   digitalWrite(S_LED, LOW);
  delay(1000);
}

void loop_switch()
{
  int state = false;
  if (digitalRead(RST_SWH) == LOW) {
    state = true;
  }
  if (state) {
    lockFunc();
    state = false;
  }
}
