
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

config = {
    'description': 'lineup',
    'author': 'Neil Seward, Mahboubeh Ahmadalinezhad',
    'author_email': 'neil.seawrd@uoit.ca, Mahboubeh.Ahmadalinezhad@uoit.ca',
    'version': '0.0.1',
    'packages': find_packages(),
    'name': 'lineup'
}

setup(**config)
