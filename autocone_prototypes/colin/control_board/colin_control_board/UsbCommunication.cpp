#include "UsbCommunication.h"

UsbCommunication::UsbCommunication(HardwareSerial &HWserial, uint32_t baudrate, uint32_t bufSize){

  pSerial = &HWserial;
  pSerial->begin(baudrate);

  bufferSize = bufSize;

  inBuffer = malloc(bufferSize * sizeof(char));
  memset (inBuffer, '\0', bufferSize);
  inBufferInd = 0;

  newData = false;
  
}

void UsbCommunication::Check(void){

  bool char_inicial = false;
  bool char_final = false;
 
  while( pSerial->available() ){
 
    char receivedChar = pSerial->read();

    if( receivedChar == '&' ){
      char_inicial = true;
    
    }else if( receivedChar == '*' ){
      newData = true;
      break;
      
    }else{
     
      if( char_inicial == true ){
        inBuffer[inBufferInd] = receivedChar;
        inBufferInd++;
      }
      
    } 
  }    
}

void UsbCommunication::ReadMessage(int32_t dataInput[], int32_t numDataInput){

  char campo[16];
  int indCampo = 0;
  int indDataInput = 0;

  for( int i = 0 ; i < bufferSize ; i++ ){

    if( inBuffer[i] == ';' ){
      int value = (int)atoi(campo);

      if( indDataInput < numDataInput ){
        dataInput[indDataInput] = value;
        indDataInput++;    
      }else{
        break;
      }

      memset(campo, '\0', sizeof(campo));
      indCampo = 0;
      
    }else if( inBuffer[i] == '*'){
      break;
      
    }else{
      campo[indCampo] = inBuffer[i];
      indCampo++;
      
    }
    
  }
 
  newData = false;
  memset (inBuffer, '\0', bufferSize);
  inBufferInd = 0;
  
}

bool UsbCommunication::HasNewData(void){
  return newData;
}

