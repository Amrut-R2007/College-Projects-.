#define BLYNK_TEMPLATE_ID "TMPL3AeZXd32J"
#define BLYNK_TEMPLATE_NAME "Waste Management"
#define BLYNK_AUTH_TOKEN "LItK32sUquNYfJt0FqfXwBYskPqx_7jl"

#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include <NTPClient.h>
#include <WiFiUdp.h>

char ssid[] = "Amrut3286";
char pass[] = "Amrut@2007";

// NTP Client Setup (IST is UTC + 5:30 -> 19800 seconds)
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 19800, 60000);

float currentFillPercent = 0.0;
float previousHourlyFill = 0.0;
bool alert85Sent = false;
int lastHour = -1;
int lastDay = -1;

void setup() {
  Serial.begin(115200); // Debug to PC terminal
  Serial2.setTimeout(5000); // Allow up to 5 seconds to catch the trailing '\n'
  Serial2.begin(115200, SERIAL_8N1, 16, 17); // UART connection with STM32 (RX=16, TX=17)
  
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
  timeClient.begin();
}

void loop() {
  Blynk.run();
  timeClient.update();
  
  // ========================================================
  // 1. READ UART DATA FROM STM32 & UPDATE REAL-TIME METRICS
  // ========================================================
  if (Serial2.available()) {
    String incoming = Serial2.readStringUntil('\n');
    incoming.trim(); // Clean trailing spaces or carriage returns
    
    if (incoming.length() > 0) {
      currentFillPercent = incoming.toFloat();
      Serial.print("Raw data from STM32: "); 
      Serial.println(currentFillPercent);
      
      // Update real-time gauge on V0 instantly
      Blynk.virtualWrite(V0, currentFillPercent);
      
      // Reactive 85% System Notification Event
      if (currentFillPercent >= 85.0 && !alert85Sent) {
        Blynk.logEvent("bin_full", "Bin is 85% full! Schedule collection.");
        alert85Sent = true;
      } else if (currentFillPercent < 20.0) {
        alert85Sent = false; // Reset threshold flag when bin gets emptied
      }

      // ==========================================
      // DYNAMIC PREDICTION LOGIC (RUNS INSTANTLY)
      // ==========================================
      float fillRatePerHour = currentFillPercent - previousHourlyFill;

      // [CRITICAL STATUS] Level is 85% or greater
      if (currentFillPercent >= 85.0) {
        Blynk.virtualWrite(V3, "CRITICAL: Immediate Pickup Req!");
      } 
      // [WARNING STATUS] Level is between 65% and 85%
      else if (currentFillPercent >= 65.0 && currentFillPercent < 85.0) {
        if (fillRatePerHour > 0) {
          float hoursToCritical = (85.0 - currentFillPercent) / fillRatePerHour;
          int hours = (int)hoursToCritical;
          int minutes = (int)((hoursToCritical - hours) * 60);
          
          char forecast_msg[40];
          if (hours > 0) {
            sprintf(forecast_msg, "Critical in %dh %dm", hours, minutes);
          } else {
            sprintf(forecast_msg, "Critical in %dm", minutes);
          }
          Blynk.virtualWrite(V3, forecast_msg);
        } else {
          Blynk.virtualWrite(V3, "HIGH LEVEL: Approaching Full");
        }
      } 
      // [MONITORING STATUS] Level is between 20% and 65%
      else if (currentFillPercent >= 20.0 && currentFillPercent < 65.0) {
        if (fillRatePerHour > 0) {
          float hoursToCritical = (85.0 - currentFillPercent) / fillRatePerHour;
          int hours = (int)hoursToCritical;
          
          char forecast_msg[40];
          sprintf(forecast_msg, "Filling: ~%dh to full", hours);
          Blynk.virtualWrite(V3, forecast_msg);
        } else {
          Blynk.virtualWrite(V3, "Monitoring: Level Stable");
        }
      } 
      // [SAFE STATUS] Level is below 20%
      else {
        Blynk.virtualWrite(V3, "No pickup needed");
      }
    }
  }

  // ========================================================
  // 2. HOURLY ANALYTICS TREND LOCKING (TIME-BASED TRACKING)
  // ========================================================
  if (timeClient.isTimeSet()) {
    int currentHour = timeClient.getHours();
    if (currentHour != lastHour) {
      lastHour = currentHour;
      
      // Log the current level onto the SuperChart trend line (V2) once an hour
      Blynk.virtualWrite(V2, currentFillPercent);
      
      // Establish the benchmark baseline for the next hour's velocity calculations
      previousHourlyFill = currentFillPercent;
    }
    
    // ========================================================
    // 3. DAILY HISTORICAL BAR CHART LOGGER
    // ========================================================
    int currentDay = timeClient.getDay(); // 0=Sunday, 1=Monday...
    if (currentDay != lastDay) {
      // Push final end-of-day peak capacity to the history bar chart (V1)
      Blynk.virtualWrite(V1, currentFillPercent); 
      lastDay = currentDay;
    }
  }
}