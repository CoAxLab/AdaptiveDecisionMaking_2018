from setuptools import setup, find_packages
import numpy as np
import os

package_data = {'ADMCode':['notebooks/*.ipynb', 'data/*.csv']}

setup(
    name='AdaptiveDecisionMaking_2018',
    version='0.0.1',
    author='Kyle Dunovan, Timothy Verstynen',
    author_email='dunovank@gmail.com',
    url='http://github.com/CoAxLab/AdaptiveDecisionMaking_2018',
    packages=['ADMCode'],
    package_data=package_data,
    description='Code and lab resources for Neural and Cognitive Models of Adaptive Decision Making course (2018)',
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'numba', 'future'],
    include_dirs = [np.get_include()],
    classifiers=[
                'Environment :: Console',
                'Operating System :: OS Independent',
                'License :: OSI Approved :: MIT License',
                'Development Status :: 3 - Alpha',
                'Programming Language :: Python',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.4',
                'Programming Language :: Python :: 3.6',
                'Topic :: Scientific/Engineering',
                ]
)
