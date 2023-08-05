#include "pir.h"
#include <Arduino.h>
#include "led.h"
#include "gsm.h"
void pirDetection()
{
  int pirValue = digitalRead(PIR);  // Read the PIR sensor value

  if (pirValue == HIGH) {
    // Motion detected
    Serial.print("Detected");
    message();
    digitalWrite(BUZZER_PIN, HIGH);
    fLED();
    delay(20000);
    digitalWrite(BUZZER_PIN, LOW);
  } else {
    digitalWrite(BUZZER_PIN, LOW);
  }

  delay(1000);  // Delay for 1 second
}
