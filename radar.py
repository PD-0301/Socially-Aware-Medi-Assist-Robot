# config_sender.py
import serial, time
cli = serial.Serial('COM8', 115200)
with open("C:\\ti\\mmwave_sdk_03_06_02_00-LTS\\packages\\ti\\demo\\xwr18xx\\mmw\\profiles\\profile_2d.cfg") as f:
    for line in f:
        if line.strip() and not line.startswith('%'):
            cli.write((line.strip() + '\n').encode())
            time.sleep(0.1)
cli.close()

