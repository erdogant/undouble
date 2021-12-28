# undouble

[![Python](https://img.shields.io/pypi/pyversions/undouble)](https://img.shields.io/pypi/pyversions/undouble)
[![PyPI Version](https://img.shields.io/pypi/v/undouble)](https://pypi.org/project/undouble/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/undouble/blob/master/LICENSE)
[![Github Forks](https://img.shields.io/github/forks/erdogant/undouble.svg)](https://github.com/erdogant/undouble/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/undouble.svg)](https://github.com/erdogant/undouble/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-blue)](https://erdogant.github.io/undouble/)
[![Downloads](https://pepy.tech/badge/undouble/month)](https://pepy.tech/project/undouble/month)
[![Downloads](https://pepy.tech/badge/undouble)](https://pepy.tech/project/undouble)
[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

* undouble is Python package

### Contents
- [Installation](#-installation)
- [Contribute](#-contribute)
- [Citation](#-citation)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install undouble from PyPI (recommended). undouble is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* A new environment can be created as following:

```bash
conda create -n env_undouble python=3.7
conda activate env_undouble
```

```bash
pip install undouble            # normal install
pip install --upgrade undouble # or update if needed
```

* Alternatively, you can install from the GitHub source:
```bash
# Directly install from github source
pip install -e git://github.com/erdogant/undouble.git@0.1.0#egg=master
pip install git+https://github.com/erdogant/undouble#egg=master
pip install git+https://github.com/erdogant/undouble

# By cloning
git clone https://github.com/erdogant/undouble.git
cd undouble
pip install -U .
```  

#### Import undouble package
```python
from undouble import Undouble
```

#### Example:
```python
df = pd.read_csv('https://github.com/erdogant/hnet/blob/master/undouble/data/example_data.csv')
model = undouble.fit(df)
G = undouble.plot(model)
```
<p align="center">
  <img src="https://github.com/erdogant/undouble/blob/master/docs/figs/fig1.png" width="600" />
  
</p>


#### References
* https://github.com/erdogant/undouble

#### Citation
Please cite in your publications if this is useful for your research (see citation).
   
### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)

### Contribute
* All kinds of contributions are welcome!
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)

### Licence
See [LICENSE](LICENSE) for details.

### Other interesting stuf
* https://ourcodeworld.com/articles/read/1006/how-to-determine-whether-2-images-are-equal-or-not-with-the-perceptual-hash-in-python
* https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
* https://github.com/JohannesBuchner/imagehash
* https://ourcodeworld.com/articles/read/1006/how-to-determine-whether-2-images-are-equal-or-not-with-the-perceptual-hash-in-python
* https://stackoverflow.com/questions/64994057/python-image-hashing
* https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34
