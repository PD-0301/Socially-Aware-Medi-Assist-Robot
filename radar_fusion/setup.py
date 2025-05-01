from setuptools import setup

package_name = 'radar_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/radar_fusion_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='priyadarshan',
    maintainer_email='ps4907@nyu.edu',
    description='Radar and Camera Sensor Fusion for Obstacle Detection',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'radar_fusion_node = radar_fusion.radar_fusion_node:main',
        ],
    },
)
