void setup() {
  Serial.begin(115200);  // USB serial communication

  // === ESC PWM on pin 25 ===
  ledcAttach(25, 0);       // Channel 0 for ESC
  ledcSetup(0, 50, 16);       // 50 Hz, 16-bit resolution

  // === Steering Servo PWM on pin 26 ===
  ledcAttach(26, 1);       // Channel 1 for Steering
  ledcSetup(1, 50, 16);       // 50 Hz, 16-bit resolution

  // Send neutral signals at startup
  ledcWrite(0, (1500 * 65535L) / 20000L);  // Stop ESC
  ledcWrite(1, (1500 * 65535L) / 20000L);  // Center steering

  Serial.println("ESP32 Motor Control Ready");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');  // Read full line

    if (cmd.startsWith("D:")) {
      int pwm = cmd.substring(2).toInt();  // Extract PWM
      int duty = (pwm * 65535L) / 20000L;
      ledcWrite(0, duty);  // Write to ESC
      Serial.printf("Drive PWM: %d\n", pwm);
    }

    else if (cmd.startsWith("S:")) {
      int pwm = cmd.substring(2).toInt();  // Extract PWM
      int duty = (pwm * 65535L) / 20000L;
      ledcWrite(1, duty);  // Write to steering
      Serial.printf("Steer PWM: %d\n", pwm);
    }
  }
}
