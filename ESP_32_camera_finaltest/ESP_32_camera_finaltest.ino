#include <Bluepad32.h>
#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <std_msgs/msg/int32.h>
#include <std_msgs/msg/int8.h>
#include <std_msgs/msg/bool.h>

// ROS2 Communication
rcl_publisher_t drive_publisher;
std_msgs__msg__Int32 drive_msg;

rcl_publisher_t steer_publisher;
std_msgs__msg__Int32 steer_msg;

rcl_publisher_t mode_publisher;
std_msgs__msg__Int8 mode_msg;

rcl_publisher_t button_publisher;
std_msgs__msg__Int8 button_msg;

rcl_subscription_t drive_subscriber;
std_msgs__msg__Int32 drive_input_msg;

rcl_subscription_t steer_subscriber;
std_msgs__msg__Int32 steer_input_msg;

rcl_subscription_t button_subscriber;
std_msgs__msg__Int8 button_input_msg;

// Radar distance subscriber
rcl_subscription_t radar_subscriber;
std_msgs__msg__Int32 radar_msg;
volatile int obstacle_distance = 9999;  // default no obstacle

// Camera detected subscriber
rcl_subscription_t camera_subscriber;
std_msgs__msg__Bool camera_msg;
volatile bool camera_detected = false;

rclc_support_t support;
rcl_allocator_t allocator;
rcl_node_t node;
rclc_executor_t executor;

ControllerPtr myController = nullptr;

// Hardware
const int ESC_PIN = 25;     
const int SERVO_PIN = 26;   

// Drive parameters
int mode;
int pre_mode;
int escMin = 1500;
int escMax = 1500;
int ESC_min_Y = 1000;
int ESC_min_B = 1350;
int ESC_min_A = 1425;
int ESC_stp_N = 1500;
int ESC_max_A = 1575;
int ESC_max_B = 1650;
int ESC_max_Y = 2000;
int drive = 1500;
int SteerX = 1500;
int pwmSteerUs = 1500;

unsigned long lastDriveInputTime = 0;
const unsigned long driveInputTimeout = 500;
bool rosDriveActive = false;

unsigned long lastSteerInputTime = 0;
const unsigned long steerInputTimeout = 500;
bool rosSteerActive = false;

// Obstacle avoidance thresholds
const int danger_distance_stop = 20;
const int danger_distance_crawl = 30;
const int danger_distance_slow = 50;

// ==================== CALLBACKS ====================

void radar_callback(const void * msgin) {
    const std_msgs__msg__Int32 * msg = (const std_msgs__msg__Int32 *)msgin;
    obstacle_distance = msg->data;
}

void camera_callback(const void * msgin) {
    const std_msgs__msg__Bool * msg = (const std_msgs__msg__Bool *)msgin;
    camera_detected = msg->data;
}

void drive_callback(const void * msgin) {
    const std_msgs__msg__Int32 * msg = (const std_msgs__msg__Int32 *)msgin;
    drive = msg->data;
    lastDriveInputTime = millis();
}

void steer_callback(const void * msgin) {
    const std_msgs__msg__Int32 * msg = (const std_msgs__msg__Int32 *)msgin;
    pwmSteerUs = msg->data;
    lastSteerInputTime = millis();
}

void button_callback(const void * msgin) {
    const std_msgs__msg__Int8 * msg = (const std_msgs__msg__Int8 *)msgin;
    uint8_t buttons = msg->data;
    if (buttons & 1) mode = 1;
    else if (buttons & 2) mode = 2;
    else if (buttons & 4) mode = 3;
    else if (buttons & 16 && buttons & 32) mode = 0;
}

// ==================== CONTROLLER HANDLERS ====================

void handleModeSelection() {
    if (!myController || !myController->isConnected()) return;
    if (myController->a()) mode = 1;
    else if (myController->b()) mode = 2;
    else if (myController->y()) mode = 3;
    else if (myController->l1() && myController->r1()) mode = 0;
    else if (myController->x()) mode = 0;

    switch (mode) {
        case 1: escMin = ESC_min_A; escMax = ESC_max_A; break;
        case 2: escMin = ESC_min_B; escMax = ESC_max_B; break;
        case 3: escMin = ESC_min_Y; escMax = ESC_max_Y; break;
        default: escMin = ESC_stp_N; escMax = ESC_stp_N; break;
    }

    if (mode != pre_mode) {
        Serial.printf("Switched to Mode %d | ESC Range: %d - %d\n", mode, escMin, escMax);
        pre_mode = mode;
    }
}

void onConnectedController(ControllerPtr ctl) {
    Serial.println("Controller connected.");
    myController = ctl;
}

void onDisconnectedController(ControllerPtr ctl) {
    Serial.println("Controller disconnected.");
    if (myController == ctl) myController = nullptr;
}

// ==================== SETUP ====================

void setup() {
    Serial.begin(115200);
    delay(2000);
    Serial.println("Starting setup...");

    set_microros_transports();
    allocator = rcl_get_default_allocator();
    rclc_support_init(&support, 0, NULL, &allocator);
    rclc_node_init_default(&node, "esp32_node", "", &support);

    rclc_publisher_init_default(&drive_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "Drive");
    rclc_publisher_init_default(&steer_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "Steer");
    rclc_publisher_init_default(&mode_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int8), "Mode");
    rclc_publisher_init_default(&button_publisher, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int8), "Button");

    rclc_subscription_init_default(&drive_subscriber, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "drive_input");
    rclc_subscription_init_default(&steer_subscriber, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "steer_input");
    rclc_subscription_init_default(&button_subscriber, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int8), "buttons_input");
    rclc_subscription_init_default(&radar_subscriber, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32), "radar_distance");
    rclc_subscription_init_default(&camera_subscriber, &node, ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool), "camera_detected");

    rclc_executor_init(&executor, &support.context, 5, &allocator);
    rclc_executor_add_subscription(&executor, &drive_subscriber, &drive_input_msg, &drive_callback, ON_NEW_DATA);
    rclc_executor_add_subscription(&executor, &steer_subscriber, &steer_input_msg, &steer_callback, ON_NEW_DATA);
    rclc_executor_add_subscription(&executor, &button_subscriber, &button_input_msg, &button_callback, ON_NEW_DATA);
    rclc_executor_add_subscription(&executor, &radar_subscriber, &radar_msg, &radar_callback, ON_NEW_DATA);
    rclc_executor_add_subscription(&executor, &camera_subscriber, &camera_msg, &camera_callback, ON_NEW_DATA);

    BP32.setup(&onConnectedController, &onDisconnectedController);
    BP32.enableVirtualDevice(false);

    ledcAttachPin(ESC_PIN, 0);
    ledcSetup(0, 50, 16);
    ledcAttachPin(SERVO_PIN, 1);
    ledcSetup(1, 50, 16);

    mode = 0;
    pre_mode = -1;
    drive = ESC_stp_N;
}

// ==================== LOOP ====================

void loop() {
    bool updated = BP32.update();
    handleModeSelection();
    rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10));
    rosDriveActive = (millis() - lastDriveInputTime) <= driveInputTimeout;
    rosSteerActive = (millis() - lastSteerInputTime) <= steerInputTimeout;

    uint8_t buttons = 0;
    if (myController) {
        if (myController->a())  buttons |= 1;
        if (myController->b())  buttons |= 2;
        if (myController->y())  buttons |= 4;
        if (myController->x())  buttons |= 8;
        if (myController->l1()) buttons |= 16;
        if (myController->r1()) buttons |= 32;
    }

    // --- Dynamic Obstacle Avoidance ---
    if (obstacle_distance < danger_distance_slow && camera_detected) {
        if (obstacle_distance < danger_distance_stop) {
            drive = 1500;
        }
        else if (obstacle_distance < danger_distance_crawl) {
            drive = map(obstacle_distance, danger_distance_stop, danger_distance_crawl, 1500, escMin + 50);
            drive = constrain(drive, 1500, escMin + 70);
        }
        else {
            drive = map(obstacle_distance, danger_distance_crawl, danger_distance_slow, escMin + 100, escMax - 100);
            drive = constrain(drive, escMin + 100, escMax - 100);
        }
    }
    // --- End Obstacle Avoidance ---

    if (updated && myController && myController->isConnected() && myController->hasData()) {
        if (!rosDriveActive) {
            int throttle = myController->throttle();
            int brake = myController->brake();

            if (brake < 1 && throttle > 5) {
                drive = map(throttle, 0, 1023, ESC_stp_N, escMax);
            } else if (brake > 5 && throttle < 1) {
                drive = map(brake, 0, 1023, ESC_stp_N, escMin);
            } else {
                drive = 1500;
            }
        }

        if (!rosSteerActive) {
            int steerX = myController->axisX();
            pwmSteerUs = map(steerX, -511, 512, 1000, 2000);
        }
    }

    int dutyThrottle = (drive * 65535L) / 20000L;
    ledcWrite(0, dutyThrottle);

    int dutySteer = (pwmSteerUs * 65535L) / 20000L;
    ledcWrite(1, dutySteer);

    drive_msg.data = drive;
    rcl_publish(&drive_publisher, &drive_msg, NULL);
    steer_msg.data = pwmSteerUs;
    rcl_publish(&steer_publisher, &steer_msg, NULL);
    button_msg.data = buttons;
    rcl_publish(&button_publisher, &button_msg, NULL);
    mode_msg.data = mode;
    rcl_publish(&mode_publisher, &mode_msg, NULL);

    Serial.printf("Mode: %d, Drive PWM: %d, Steer PWM: %d, Obstacle: %d cm, Camera: %d\n", 
                  mode, drive, pwmSteerUs, obstacle_distance, camera_detected);
}
