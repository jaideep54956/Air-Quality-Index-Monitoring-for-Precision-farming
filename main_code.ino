#include <LiquidCrystal.h>
#include "DHT.h"
#include<Servo.h>
#define DHTPIN A3
int sts=1;
const int rs = 8, en = 9, d4 = 10, d5 = 11, d6 = 12, d7 = 13;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

Servo gate;
#define s1 A0
#define s2 A1
#define s3 A2
#define DHTTYPE DHT11 
int s1val,s2val,s3val,t,h;
int cnt=0;
int buz=A4;
DHT dht(DHTPIN, DHTTYPE);
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); 
  lcd.begin(16, 2);
  pinMode(buz,OUTPUT);
  dht.begin();
  lcd.print("   WELCOME");
  gate.attach(7);
  gate.write(0);
  delay(30000);
}

void loop() {
  // put your main code here, to run repeatedly:

  s1val=analogRead(s1)/10.23;
  s2val=analogRead(s2)/10.23;
  s3val=analogRead(s3)/10.23;
  t = dht.readTemperature();
  h = dht.readHumidity();
  lcd.clear();
  lcd.print("M:"+ String(s1val) + " C:"+ String(s2val)+ " S:"+ String(s3val));
  lcd.setCursor(0,1);
 lcd.print("T:"+ String(t) + " H:"+ String(h));
 String str=String(s1val)+","+ String(s2val)+","+ String(s3val)+","+ String(t)+","+ String(h);
 if(Serial.available())
 {
   sts=Serial.read();
 }

  if(sts=='1')
  {
    lcd.setCursor(10,1);
    lcd.print("GOOD");
  }

   if(sts=='2')
  {
    lcd.setCursor(10,1);
    lcd.print("MODRTE");
  }

   if(sts=='3')
  {
     digitalWrite(buz,1);
    lcd.setCursor(10,1);
    lcd.print("BAD");
    delay(300);
    digitalWrite(buz,0);
  }

  if(t>35)
  {
    gate.write(90);
    
  }
  else
  {
     gate.write(0);
  }
  cnt=cnt+1;
  delay(1000);
  if(cnt>15)
  {
    cnt=0;
    Serial.println(str);
    delay(1000);
  }
}
