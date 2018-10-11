
#ifndef ServoMotor_h
#define ServoMotor_h

#include <Arduino.h>
#include "Dynamixel_Serial.h"

class ServoMotor{

  public:

    ServoMotor(HardwareSerial &HWserial, uint32_t baudrate, uint8_t id, uint8_t servoControlPin);

    int32_t GetPosition(void);
    void SetPosition(int32_t pos);

    uint32_t ReadServoPosition(void);


  private:
    
    uint8_t servoId;
    uint32_t servoBaudrate;
    int32_t servoCWLimitAngle;
    int32_t servoCCWLimitAngle;
    int32_t servoSpeed;
    uint8_t servoControlPin;

    int32_t servoCurrentAngle;
    
    
    int32_t currentPosition;
    int32_t maxPosition;

    

    
    
};

#endif
