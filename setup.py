import os
from setuptools import setup, find_packages

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

setup(
    name = "UiT_asimoc",
    version = "1.0",
    author = "Qiang Wang, Clément Stouls, Catherine Taelman, Jozef Rusin, Johannes Lohse",
    author_email = "jlo031@uit.no",
    description = ("ice/water deep learning classifier"),
    license = "cc BY NC ND",
    long_description=read('README.md'),
    packages = find_packages(where='src'),
    package_dir = {'': 'src'},
    package_data = {'': ['*.xml', '.env']},
    entry_points = {
        'console_scripts': [
        ]
    },
    include_package_data=True,
)
