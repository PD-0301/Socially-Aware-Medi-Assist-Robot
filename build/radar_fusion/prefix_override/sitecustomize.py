import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/priyadarshan/Socially-Aware-Medi-Assist-Robot/install/radar_fusion'
