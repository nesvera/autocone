#include "ServoMotor.h"

#define SERVO_ControlPin 2            // Control pin of buffer chip, NOTE: this does not matter becasue we are not using a half to full contorl buffer.

ServoMotor::ServoMotor(HardwareSerial &HWserial, uint32_t baudrate, uint8_t id, uint8_t controlPin){
  servoId = id;
  servoBaudrate = baudrate;
  servoControlPin = controlPin;

  servoCWLimitAngle = 2733;
  servoCCWLimitAngle = 3748;

  servoSpeed = 1023;

  maxPosition = 100;
  currentPosition = 0;

  servoCurrentAngle = map(currentPosition, -maxPosition, maxPosition, servoCWLimitAngle, servoCCWLimitAngle);

  Dynamixel.begin(servoBaudrate);                              
  Dynamixel.setDirectionPin(servoControlPin);                        
  Dynamixel.setMode(servoBaudrate, SERVO, servoCWLimitAngle, servoCCWLimitAngle);   
  
}

int32_t ServoMotor::GetPosition(void){
  return currentPosition;
}

void ServoMotor::SetPosition(int32_t pos){

  currentPosition = pos;
  servoCurrentAngle = map(pos, -maxPosition, maxPosition, servoCWLimitAngle, servoCCWLimitAngle);

  if( servoCurrentAngle < servoCWLimitAngle ){
    servoCurrentAngle = servoCWLimitAngle;
    
  }else if( servoCurrentAngle > servoCCWLimitAngle ){
    servoCurrentAngle = servoCCWLimitAngle;
    
  }

  // Move servo to goalPosition with velocity
  Dynamixel.servo(servoId, servoCurrentAngle, servoSpeed);
  
}

uint32_t ServoMotor::ReadServoPosition(void){
  return Dynamixel.readPosition(servoId);
}

