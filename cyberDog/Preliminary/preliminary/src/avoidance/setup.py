from setuptools import setup

package_name = 'avoidance'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mi',
    maintainer_email='mi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'stand = avoidance.stand:main',
            'stand_sit = avoidance.stand_sit:main',
            'sensor = avoidance.sensor:main',
            'walk = avoidance.walk:main',
            'avoid = avoidance.avoid:main',
            'radar = avoidance.radar:main',
            'ultrasonic = avoidance.ultrasonic:main',
            'avoid_jez = avoidance.avoid_jez:main',
        ],
    },
)
