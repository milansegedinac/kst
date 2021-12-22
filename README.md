# Knowledge Space Theory

KST is an open source software library with Python implementations of basic Knowledge Space Theory algorithms.

## Usage

### Installation
You can either clone the project or download a distribution [file](./dist/learning_spaces-0.2.0-py3-none-any.whl) and run command:
`pip install /path-to-downloaded-file/learning_spaces-0.2.0-py3-none-any.whl`

### Setup in Python
KST requires installed Python 3.9. It is recommended to use the library in a separate virtual environment. A brief and practical introduction to virtual environments can be found on the following [link](https://docs.python-guide.org/dev/virtualenvs/).
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
>>> from learning_spaces.kst import iita
>>> data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
>>> response = iita(data_frame, v=1)
>>> print(response)
{'diff': array([ 0.18518519,  0.16666667,  0.21296296]), 'implications': [(0, 1), (0, 2), (2, 0), (2, 1)], 'error.rate': 0.5, 'selection.set.index': 1, 'v': 1}
```

### Setup in a browser
KST can be run in a browser environment, without need for Python server. We use [Pyodide](https://github.com/pyodide/pyodide) which brings the Python runtime to the browser via WebAssembly.

Full Example (open console to see the result):
```html
<!DOCTYPE html>
  <html>
  <head>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.18.1/full/pyodide.js"></script>
  </head>
  <body>
    <script>
      let pyodide;

      async function init() {
        pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.18.1/full/" });
        await pyodide.loadPackage('micropip');
        await pyodide.runPythonAsync(`
          from micropip import install
          await install('https://raw.githubusercontent.com/milansegedinac/kst/master/dist/learning_spaces-0.2.0-py3-none-any.whl')
        `);
      }

      async function run() {
        await pyodide.runPython(`
          import pandas as pd
          from learning_spaces.kst import iita
          data_frame = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 0], 'c': [0, 1, 1]})
          response = iita(data_frame, v=1)
        `);

        const response = pyodide.globals.get('response').toJs()
        console.log(response)
      }

      (async () => {
        await init()
        await run()
      })();
    </script>
  </body>
</html>
```