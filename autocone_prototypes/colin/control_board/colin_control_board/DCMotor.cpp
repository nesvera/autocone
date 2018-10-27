#include "DCMotor.h"
#include "Arduino.h"

DCMotor::DCMotor(){

  noInterrupts();

  topSpeed = 100;
  maxVelocity = 100;
  direction = 0;
  velocity = 0;

  pinMode(R_IS, INPUT);
  pinMode(L_IS, INPUT);
  pinMode(R_EN, OUTPUT);  digitalWrite(R_EN, HIGH);
  pinMode(L_EN, OUTPUT);  digitalWrite(L_EN, HIGH);
  pinMode(RPWM, OUTPUT);  digitalWrite(RPWM, LOW);
  pinMode(LPWM, OUTPUT);  digitalWrite(LPWM, LOW);

  // Configure timer to FAST PWM with TOP = OCRnA
  TCCR5A |= (1 << COM5A1);
  TCCR5A |= (1 << WGM51);
  TCCR5A |= (1 << WGM50);
  TCCR5B |= (1 << WGM53);
  TCCR5B |= (1 << WGM52);

  // Configure PWM Frequency

  // Prescaler = 1
  TCCR5B &= ~(1 << CS52);     // CS52 = 0
  TCCR5B &= ~(1 << CS51);     // CS51 = 0
  TCCR5B |= (1 << CS50);      // CS50 = 1

  timerTOP = 800;

  OCR5AH = timerTOP >> 8;
  OCR5AL = timerTOP & 0xFF;

  interrupts();
  
}

uint8_t DCMotor::GetDutyCycle(void){
  return goalDutyCycle;  
}

void DCMotor::SetDutyCycle(uint8_t dutyCycle){
  goalDutyCycle = dutyCycle;
}

void DCMotor::SetVelocity(int velocity){

  bool dir;
  uint8_t vel;

  if( abs(velocity) > maxVelocity ){
    vel = maxVelocity;
  }else{
    vel = abs(velocity);
  }

  // Velocity to dutyCycle
  float timerVal = map(vel, 0, topSpeed, 0, timerTOP);
  Serial.println(timerVal);

  // FORWARD
  if( velocity > 0 ){
    
    analogWrite(LPWM, 0);
    analogWrite(RPWM, timerVal);
    
    dir = 0;


  // BACKWARD
  }else{

    analogWrite(RPWM, 0);
    analogWrite(LPWM, timerVal);
    
    dir = 1;
  
  }


}

void DCMotor::Forward(uint16_t dutyCycle){
  
}

void DCMotor::Barward(uint16_t dutyCycle){
  
}

int DCMotor::GetMaxVelocity(void){
  return maxVelocity;
}

void DCMotor::SetMaxVelocity(unsigned int maxVel){
  if( maxVel > 0 && maxVel < UINT32_MAX ){
    maxVelocity = maxVel;
  }
}

