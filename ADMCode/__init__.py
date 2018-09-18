"""
Code and lab resources for "Neural and Cognitive Models of
Adaptive Decision Making" course (2018)

Authors:
  CoAxLab at github.com

"""

from __future__ import print_function
import os
import sys
from glob import glob

modules = glob(os.path.dirname(__file__) + "/*.py")
__all__ = [os.path.basename(f)[:-3] for f in modules]

major = 0
minor = 1
patch = 0
__version__ = '.'.join([str(v) for v in [major, minor, patch]])

# path to local site-packages/jupyterthemes
package_dir = os.path.dirname(os.path.realpath(__file__))
