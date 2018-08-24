# Knowledge Space Theory

KST is an open source software library with Python implementations of basic Knowledge Space Theory algorithms.

## Usage

KST requires installed Python 3.6. It is recommended to use the library in a separate virtual environment. A brief and practical introduction to virtual environments can be found on the following [link](https://docs.python-guide.org/dev/virtualenvs/).
First, a virtual environment should be created.
```
mkvirtualenv kst
```
After creating a virtual environment, you should install the requirements.
```
pip install -r requirements.txt
```
After that, the library can be used.
```python
>>> import pandas as pd
>>> import numpy as np
>>> import sys
>>> sys.path.append('learning_spaces/')
>>> from learning_spaces.kst import iita
>>> data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
>>> response = iita(data_frame, v=1)
>>> print(response)
{'diff': array([ 0.18518519,  0.16666667,  0.21296296]), 'implications': [(0, 1), (0, 2), (2, 0), (2, 1)], 'error.rate': 0.5, 'selection.set.index': 1, 'v': 1}
```
