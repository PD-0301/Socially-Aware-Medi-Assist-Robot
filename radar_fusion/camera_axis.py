import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Intrinsic matrix (from user's camera calibration)
K = np.array([[386.59464185, 0.0, 385.10781334],
              [0.0, 379.00255236, 274.90390815],
              [0.0, 0.0, 1.0]])

# Camera coordinate axes (unit vectors)
origin = np.array([0, 0, 0])
x_axis = np.array([1, 0, 0])  # Red
y_axis = np.array([0, 1, 0])  # Green
z_axis = np.array([0, 0, 1])  # Blue

# Prepare the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(*origin, *x_axis, color='r', label='X (Right)')
ax.quiver(*origin, *y_axis, color='g', label='Y (Down)')
ax.quiver(*origin, *z_axis, color='b', label='Z (Forward)')

# Annotate the plot
ax.text(*x_axis, 'X', color='r')
ax.text(*y_axis, 'Y', color='g')
ax.text(*z_axis, 'Z', color='b')

# Formatting
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel('X-axis (Right)')
ax.set_ylabel('Y-axis (Down)')
ax.set_zlabel('Z-axis (Forward)')
ax.set_title('Camera Coordinate Axes (OpenCV Convention)')
ax.legend()
plt.tight_layout()
plt.show()
