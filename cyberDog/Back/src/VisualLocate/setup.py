from setuptools import setup

package_name = 'VisualLocate'

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
    maintainer_email='liujk22@mails.tsinghua.edu.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'walk = VisualLocate.walk:main',
            'stand = VisualLocate.stand:main',
            'get_image = VisualLocate.get_image:main',
        ],
    },
)
