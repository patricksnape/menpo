
<p align="center">
  <img src="menpo-logo.png" alt="menpo" width="30%"></center>
  <br><br>
  <a href="https://pypi.python.org/pypi/menpo"><img src="http://img.shields.io/pypi/v/menpo.svg?style=flat" alt="PyPI Release"/></a>
  <a href="https://github.com/menpo/menpo/blob/master/LICENSE.txt"><img src="http://img.shields.io/badge/License-BSD-green.svg" alt="BSD License"/></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <br>
  <img src="https://img.shields.io/badge/Python-3.6-green.svg" alt="Python 3.6 Support"/>
  <img src="https://img.shields.io/badge/Python-3.7-green.svg" alt="Python 3.7 Support"/>
  <img src="https://img.shields.io/badge/Python-3.8-green.svg" alt="Python 3.8 Support"/>
  <img src="https://img.shields.io/badge/Python-3.9-green.svg" alt="Python 3.9 Support"/>
</p>


Menpo. The Menpo Project Python package for handling annotated data.
====================================================================
What is Menpo?
--------------
Menpo is a [Menpo Project](http://www.menpo.org/) package designed from
the ground up to make importing, manipulating and
visualizing image and mesh data as simple as possible. In particular,
we focus on **annotated** data which is common within the fields of Machine
Learning and Computer Vision. All core types are `Landmarkable` and
visualizing these landmarks is very simple. Since landmarks are first class
citizens within Menpo, it makes tasks like masking images, cropping images
inside landmarks and aligning images very simple.

> Menpo were facial armours which covered all or part of the face and provided
> a way to secure the top-heavy kabuto (helmet). The Shinobi-no-o (chin cord)
> of the kabuto would be tied under the chin of the menpo. There were small
> hooks called ori-kugi or posts called odome located on various places to
> help secure the kabuto's chin cord.
>
> --- Wikipedia, Menpo

Installation
------------
Here in the Menpo Team, we are firm believers in making installation as simple
as possible. Unfortunately, we are a complex project that relies on satisfying
a number of complex 3rd party library dependencies. The default Python packing
environment does not make this an easy task. Therefore, we evangelise the use
of the conda ecosystem, provided by
[Anaconda](https://store.continuum.io/cshop/anaconda/). In order to make things
as simple as possible, we suggest that you use conda too! To try and persuade
you, go to the [Menpo website](http://www.menpo.io/installation/) to find
installation instructions for all major platforms.

If you feel strongly about using Menpo with the most commonly used Python
package management system, `pip`, then you should be able to install
Menpo as follows:

```
> pip install menpo
```

We strongly advocate the use of [conda](http://conda.pydata.org/docs/) which does
not require compilation for installing Menpo or it's dependencies such as Numpy, 
SciPy or Matplotlib. Installation via `conda` is as simple as

```
> conda install -c conda-forge menpo
```

#### Build Status
And has the added benefit of installing a number of commonly used scientific
packages such as SciPy and Numpy as Menpo also makes use of these packages.

|  CI Host |        OS        |                      Build Status                                     |
|:--------:|:----------------:|:---------------------------------------------------------------------:|
| CircleCI | linux/amd64      | [![menpo](https://circleci.com/gh/menpo/menpo.svg?style=svg)](<LINK>) |

Usage
-----
Menpo makes extensive use of Jupyter Notebooks to explain functionality of the
package. These Notebooks are hosted in the
[menpo/menpo-notebooks](https://github.com/menpo/menpo-notebooks) repository.
We strongly suggest that after installation you:

  1. Download the [latest version of the notebooks][notebooks_gh]
  2. Conda install Jupyter notebook and IPython: `conda install jupyter ipython notebook`
  3. Run `jupyter notebook`
  4. Play around with the notebooks.

[notebooks_gh]: https://github.com/menpo/menpo-notebooks/releases

Want to get a feel for Menpo without installing anything? You can browse the
notebooks straight from the [menpo website](http://www.menpo.io/notebooks.html).

Other Menpo projects
--------------------
Menpo is designed to be a core library for implementing algorithms within
the Machine Learning and Computer Vision fields. For example, we have developed
a number of more specific libraries that rely on the core components of Menpo:

  - [menpofit][mf_gh]: Implementations of state-of-the-art deformable modelling
    algorithms including Active Appearance Models, Constrained Local Models
    and the Supervised Descent Method.
  - [menpo3d][m3d_gh]: Useful tools for handling 3D mesh data including
    visualization and an OpenGL rasterizer. The requirements of this package
    are complex and really benefit from the use of conda!
  - [menpodetect][md_gh]: A package that wraps existing sources of object
    detection. The core project is under a BSD license, but since other projects
    are wrapped, they may not be compatible with this BSD license. Therefore,
    we urge caution be taken when interacting with this library for
    non-academic purposes.

[mf_gh]: https://github.com/menpo/menpofit
[m3d_gh]: https://github.com/menpo/menpo3d
[md_gh]: https://github.com/menpo/menpodetect

Documentation
-------------
See our documentation on [ReadTheDocs](http://menpo.readthedocs.org)

Testing
-------
We use [pytest](http://pytest.org/en/latest/) for unit tests.

After installing `pytest`, `mock` and `pytest-mock`, running

    >> pytest .

from the top of the repository will run all of the unit tests.

Some small parts of Menpo are only available if the user has some optional
dependency installed. These are:

- 3D viewing methods, only available if `menpo3d` is installed
- `menpo.feature.dsift` only available if `cyvlfeat` is installed
- Some warping unit tests are only available if `opencv` is installed
