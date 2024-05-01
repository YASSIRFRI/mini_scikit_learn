from setuptools import setup, find_packages

setup(
    name='mini_scikit_learn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    author='Yassir Fri',
    author_email='your.email@example.com',
    description='A mini scikit-learn clone package',
    url='https://github.com/YASSIRFRI/mini_scikit_learn',
)
