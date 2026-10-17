from glob import glob

from setuptools import setup

package_name = 'agri_ugv_perception_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Abdullah Bin Shahid',
    maintainer_email='abshahidthe01@gmail.com',
    description='Camera pictures for crop/weed perception from the agri-ugv-sim field.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'make_cutouts = agri_ugv_perception_sim.make_cutouts:main',
            'make_field_extras = agri_ugv_perception_sim.make_field_extras:main',
            'make_soil_texture = agri_ugv_perception_sim.make_soil_texture:main',
        ],
    },
)
