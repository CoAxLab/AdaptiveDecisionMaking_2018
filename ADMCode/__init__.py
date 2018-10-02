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
minor = 3
patch = 1
__version__ = '.'.join([str(v) for v in [major, minor, patch]])

_package_dir = os.path.dirname(os.path.realpath(__file__))

def style_notebook():
    from IPython.core.display import HTML
    _styles_dir = os.path.join(_package_dir, 'styles')
    style = os.path.join(_styles_dir, 'custom.css')
    csscontent = open(style, "r").read()
    return HTML(csscontent)
