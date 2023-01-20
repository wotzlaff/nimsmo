# Installation

## Installation of nim package

```
nimble install
```

Run example:

```
nim r -d:release examples/smo_test.nim
```

## Installation of python module

### from PyPi
```
pip install nimsmo
```

### from source
First, install dependencies:
```
conda install poetry
```


```
poetry shell
poetry install
```

Run example:

```
python examples/smo_test.py
```
