# Copyright (C) 2022-Mattia G. Bergomi, Massimo Ferri, Antonella Tavaglione,
# Lorenzo Zuffi
#
# hubpersistence is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please send an email (mattiagbergomi@gmail.com).
#

import os
import sys
from distutils.sysconfig import get_python_lib
from setuptools import find_packages, setup
import subprocess
import pathlib

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (2, 7)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of hubpersistence requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

requirements = (pathlib.Path(__file__).parent / "requirements.txt").read_text().splitlines()
EXCLUDE_FROM_PACKAGES = []

setup(
    name='hubpersistence',
    version='0.0.0-prealpha',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='',
    author='Mattia G. Bergomi, Massimo Ferri, Antonella Tavaglione, Lorenzo Zuffi',
    author_email='mattiagbergomi@gmail.com',
    description=(''),
    license='GNU General Public License v3 or later (GPLv3+)',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=requirements,
    entry_points={},
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'
        'Topic :: Scientific/Engineering :: Machine Learning',
        'Topic :: Scientific/Engineering :: Data Analysis',
        'Topic :: Scientific/Engineering :: Topology',
    ],
)
