
#ifndef UsbCommunication_h
#define UsbCommunication_h

#include <Arduino.h>

class UsbCommunication{

  public:

    UsbCommunication(HardwareSerial &HWserial, uint32_t baudrate, uint32_t bufSize);

    void Check(void);
    void ReadMessage(int32_t dataInput[], int32_t numDataInput); 

    bool HasNewData(void);
    

  private:

    HardwareSerial *pSerial;

    bool newData;

    uint32_t bufferSize;
    char *inBuffer;
    uint32_t inBufferInd;
      
    
};

#endif
