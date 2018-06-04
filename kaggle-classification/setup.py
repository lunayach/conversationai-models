from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=1.7.0', 'tflearn>=0.3.2', 'Keras==2.1.5', 'h5py==2.7.1', 'comet-ml==1.0.8', 'tensorflow_hub>=0.1.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='tflearn.'
)

setup(
    name='keras_trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='tflearn.'
)
