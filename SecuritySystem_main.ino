//vcc -5v, grnd-grnd,cli-6,reset-8,dat-7

#include <virtuabotixRTC.h>
#include "esp32.h"
#include "fingerprint.h"
#include "gsm.h"
#include "pins.h"
#include "pir.h"
#include "wifi.h"
#include "switch.h"
#include "led.h"
//rtc

virtuabotixRTC myRTC(CLK_PIN, DAT_PIN, RST_PIN);

void setup()  {
  Serial.begin(115200);
  // Set the current date, and time in the following format:
  // seconds, minutes, hours, day of the week, day of the month, month, year
  myRTC.setDS1302Time(0, 0, 8, 5, 14,7,2023);
  pinMode(PIR, INPUT);  // Set the PIR pin as input
  pinMode(LOCK_PIN,OUTPUT);
  pinMode(S_LED,OUTPUT);
  pinMode(F_LED,OUTPUT);
  pinMode(BUZZER_PIN,OUTPUT);
  pinMode(RST_SWH,INPUT_PULLUP);
  fingerprint_setup();
  wifi_setup();
  gsm_setup();
}

void loop()  {
  // This allows for the update of variables for time or accessing the individual elements.
  myRTC.updateTime();
  
  Serial.print("Current Date / Time: ");
  Serial.print(myRTC.dayofmonth);
  Serial.print("/");
  Serial.print(myRTC.month);
  Serial.print("/");
  Serial.print(myRTC.year);
  Serial.print("  ");
  Serial.print(myRTC.hours);
  Serial.print(":");
  Serial.print(myRTC.minutes);
  Serial.print(":");
  Serial.println(myRTC.seconds);
  if(myRTC.hours == 8 && myRTC.minutes >=0 && myRTC.minutes<=11){ 
    // Serial.println("RTC IF IS WORKING !!!!!!!!!");
    if(scan_finger()==1 && getVal()=="1"){
       Serial.println("if working");
      digitalWrite(LOCK_PIN,HIGH);
      digitalWrite(S_LED, HIGH);
      delay(20000);
      digitalWrite(S_LED, LOW);
      digitalWrite(LOCK_PIN,LOW);
      delay(2000);
      
    }
  }else{
    pirDetection();
    
  }

  // Delay so the program doesn't print non-stop
  delay(1000);

 
}