""" Setup script for instclf. """
from setuptools import setup

import imp

version = imp.load_source('instclf.version', 'instclf/version.py')

if __name__ == "__main__":
    setup(
        name='instclf',
        version=version.version,
        description='Musical Instrument Classifier',
        author='Hanna Yip',
        author_email='hmyip1@gmail.com',
        url='https://github.com/hmyip1/instclf',
        download_url='https://github.com/hmyip1/instclf/releases',
        packages=['instclf'],
        package_data={'instclf': []},
        long_description="""Musical instrument classifier""",
        keywords='audio instrument classifier instclf',
        license='MIT',
        install_requires=[
            "numpy",
            "scikit-learn",
            "librosa",
            "sox",
            "matplotlib",
            "scipy"
        ],
        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        }
    )