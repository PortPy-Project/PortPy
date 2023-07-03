from setuptools import setup, find_packages


def _get_portpy_photon_version():
    with open('portpy/__init__.py') as fp:
        for line in fp:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)  # pylint: disable=exec-used
                return g['__version__']
        raise ValueError('`__version__` not defined in `portpy/__init__.py`')


_VERSION = _get_portpy_photon_version()
STATUSES = [
    "1 - Planning",
    "2 - Pre-Alpha",
    "3 - Alpha",
    "4 - Beta",
    "5 - Production/Stable",
    "6 - Mature",
    "7 - Inactive"
]

setup(
    name='portpy',
    version=_VERSION,
    url='https://github.com/PortPy-Project/PortPy',
    license='Apache License, Version 2.0',
    packages=find_packages(exclude=["examples*"]),
    author='Gourav Jhanwar, Mojtaba Tefagh, Vicki Taasti, Seppo Tuomaala, Saad Nadeem, Masoud Zarepisheh',
    author_email="jhanwarg@mskcc.struct, mtefagh@acm.struct, vicki.taasti@maastro.nl, tuomaals@mskcc.struct, nadeems@mskcc.struct, zarepism@mskcc.struct",
    description='First open-source radiation treatment planning system in Python',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    python_requires='>=3.7',
    include_package_data=True,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    install_requires=[
        "cvxpy>=1.1.18",
        "ecos>=2.0.10",
        "h5py>=3.6.0",
        "Mosek>=9.3.14",
        "natsort>=8.1.0",
        "numpy>=1.15",
        "osqp>=0.4.1",
        "pandas>=1.1.5",
        "python-dateutil>=2.8.2",
        "pytz>=2022.1",
        "qdldl>=0.1.5",
        "scipy>=1.5.4",
        "scs>=3.2.0",
        "six>=1.16.0",
        "matplotlib>=3.5.3",
        "Shapely>=1.8.4",
        "SimpleITK>=2.0.2",
        "tabulate>=0.9.0",
        "typing>=3.7.4.3",
        "jinja2>=3.0.1",
        "typing-extensions>=3.10.0.0",
        "scikit-image>=0.17.2",
        "patchify>=0.2.3",
        "pydicom>=2.2.0",
    ],

)
