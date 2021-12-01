from setuptools import setup, find_packages

setup(
    name='learning_spaces',
    version='0.2.0',
    description='Knowledge Space Theory',
    url='https://github.com/milansegedinac/kst',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'pydot', 'matplotlib']
)
