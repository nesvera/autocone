#include "DCMotor.h"
#include "ServoMotor.h"
#include "UsbCommunication.h"

#define R_EN 42
#define L_EN 43
#define RPWM 44
#define LPWM 45

#define DYN_CTL_PIN 4

#define HALLSENSORPIN 3

DCMotor *motor;
ServoMotor *steering;
UsbCommunication *usbComm;

// Data from Jetson
uint8_t numDataInput = 4;
int32_t inputData[4];

// Control variables from jetson
int32_t cmdThrothle = 0;       
int32_t cmdSteering = 0;
bool jetsonStop = true;                         // stop = true -> panic button

// Control variables from radio
bool radioStop = true;

// State variables
float approxSpeed = 0.0;
volatile uint32_t hallSensorCounter = 0;
float accelX = 0.0;
float accelY = 0.0;
float accelZ = 0.0;
float gyroRoll = 0.0;
float gyroPitch = 0.0;
float gyroYaw = 0.0;

void setup() {
  // put your setup code here, to run once:

  // Configure hall effect sensor interruption
  pinMode( HALLSENSORPIN, INPUT_PULLUP );
  attachInterrupt( digitalPinToInterrupt(HALLSENSORPIN), hallSensorHandler, FALLING);  

  // Initialize rear motor
  motor = new DCMotor();
  //motor->SetMaxVelocity(50);
  motor->SetVelocity(0);
  
  delay(100);

  // Initialize steering motor
  steering = new ServoMotor(Serial2, 1000000, 1, DYN_CTL_PIN);
  steering->SetPosition(0);

  // Initialize usb communication
  usbComm = new UsbCommunication(Serial, 115200, 16);

  digitalWrite(13, OUTPUT);

}

int i = 0;
long unsigned int lastSentMsg = 0;

void loop() {

  // Check if receive data from jetson
  usbComm->Check();

  if( usbComm->HasNewData() ){
    usbComm->ReadMessage( inputData, numDataInput);

    int velocidade = inputData[0];
    //Serial.println(velocidade);
    motor->SetVelocity(velocidade);
    
    //Serial.println(inputData[1]);
    steering->SetPosition(inputData[1]);
    
  }    
}

// Counts front wheel rotation
void hallSensorHandler(){
  hallSensorCounter++;
}

