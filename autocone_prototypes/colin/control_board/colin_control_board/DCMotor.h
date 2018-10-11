#ifndef DCMotor_h
#define DCMotor_h

#include "Arduino.h"

#define R_IS A14
#define L_IS A15
#define R_EN 42
#define L_EN 43
#define RPWM 44
#define LPWM 45

class DCMotor{

  public:

    DCMotor();

    uint8_t GetDutyCycle(void);
    void SetDutyCycle(uint8_t dutyCycle);

    int GetVelocity(void);
    void SetVelocity(int velocity);

    int GetMaxVelocity(void);
    void SetMaxVelocity(unsigned int maxVel);

    void Brake();
    void Forward(uint16_t dutyCycle);
    void Barward(uint16_t dutyCycle);


  private:
    uint8_t currentDutyCycle;
    uint8_t goalDutyCycle;

    uint16_t timerTOP;

    int topSpeed;
    int maxVelocity;

    bool direction;               // 0 = forward   1 = backward
    unsigned int velocity;
    
};

#endif
