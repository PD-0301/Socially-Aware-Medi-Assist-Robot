import serial
import time

# CLI port (usually ttyACM0 for AWR1843 on Ubuntu)
cli = serial.Serial('/dev/ttyACM0', 115200)

# Path to the radar config file (your full path)
cfg_path = "/home/priyadarshan/ti/mmwave_sdk_03_06_02_00-LTS/packages/ti/demo/xwr18xx/mmw/profiles/profile_2d.cfg"

# Send each line from the config file
with open(cfg_path) as f:
    for line in f:
        if line.strip() and not line.startswith('%'):
            cli.write((line.strip() + '\n').encode())
            print(f"📤 Sent: {line.strip()}")
            time.sleep(0.1)

cli.close()
print("✅ Config sent successfully.")
