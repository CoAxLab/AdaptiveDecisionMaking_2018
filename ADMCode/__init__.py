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
minor = 4
patch = 0
__version__ = '.'.join([str(v) for v in [major, minor, patch]])

_package_dir = os.path.dirname(os.path.realpath(__file__))

def style_notebook():
    from IPython.core.display import HTML
    _styles_dir = os.path.join(_package_dir, 'styles')
    style = os.path.join(_styles_dir, 'custom.css')
    csscontent = open(style, "r").read()
    return HTML(csscontent)


def load_attractor_animation():
    import io, base64
    from IPython.display import HTML
    _examples_dir = os.path.join(_package_dir, '../notebooks/images')
    mov_fpath = os.path.join(_examples_dir, 'attractor.mp4')
    video = io.open(mov_fpath, 'r+b').read()
    encoded = base64.b64encode(video)
    data='''<video width="80%" alt="test" loop=1 controls> <source src="data:video/mp4; base64,{0}" type="video/mp4" /> </video>'''.format(encoded.decode('ascii'))
    return HTML(data=data)
