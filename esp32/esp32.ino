void setup() {
  Serial.begin(115200);

  // ESC PWM setup (pin 25)
  ledcAttachPin(25, 0);
  ledcSetup(0, 50, 16);  // 50 Hz, 16-bit

  // Steering PWM setup (pin 26)
  ledcAttachPin(26, 1);
  ledcSetup(1, 50, 16);

  // Send neutral signal at startup
  ledcWrite(0, (1500 * 65535L) / 20000L);  // ESC
  ledcWrite(1, (1500 * 65535L) / 20000L);  // Servo
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');

    if (cmd.startsWith("D:")) {
      int pwm = cmd.substring(2).toInt();
      int duty = (pwm * 65535L) / 20000L;
      ledcWrite(0, duty);  // ESC
    }

    else if (cmd.startsWith("S:")) {
      int pwm = cmd.substring(2).toInt();
      int duty = (pwm * 65535L) / 20000L;
      ledcWrite(1, duty);  // Steering
    }
  }
}
