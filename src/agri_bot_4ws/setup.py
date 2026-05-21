import os
from glob import glob
from setuptools import setup

package_name = 'agri_bot_4ws'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roboi',
    maintainer_email='jongkweanlee@gmail.com',
    description='4WS kinematics controller',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
	    'row_detector = agri_bot_4ws.row_detector:main',
	    'corridor_detector = agri_bot_4ws.corridor_detector:main',
	    'row_follower = agri_bot_4ws.row_follower:main',
            'four_ws_controller  = agri_bot_4ws.four_ws_controller:main',
            'odometry_publisher  = agri_bot_4ws.odometry_publisher:main',
        ],
    },
)
