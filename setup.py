from setuptools import setup, find_packages

setup(
    name='kst',
    version='0.1.0',
    description='Knowledge Space Theory',
    url='https://github.com/milansegedinac/kst',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'pydot', 'matplotlib']
)
