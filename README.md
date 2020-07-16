# pybnbowtie

pybnbowtie is a library for mapping bow-tie analysis to Bayesian networks.


## Dependencies

pybnbowtie has the following dependencies:

- pgmpy (with its own dependencies)
- treelib
- matploblib

Enable usage of jupyter (see also notes below)
- ipykernel

### Installation of dependencies from pgmpy

- networkx
- numpy
- scipy
- pandas
- pyparsing
- torch
- statsmodels
- tqdm
- joblib
- pgmpy
- treelib

And perhaps for jupyter also
- ipykernel
- matplotlib


## Installation

To install pybnbowtie from source code:

```
git clone https://github.com/zurheide/pybnbowtie.git
cd pybnbowtie
pip install -r requirements.txt
python setup.py install
```


## jupyter

If pipenv is used the environment has to be installed in jupyter.
Howto was found here: https://stackoverflow.com/questions/47295871/is-there-a-way-to-use-pipenv-with-jupyter-notebook

run ``python -m ipykernel install --user --name=my-virtualenv-name`` before usage of jupyter:
```
$ pipenv shell
$ python -m ipykernel install --user --name=my-virtualenv-name
$ jupyter notebook
```

Afterwards select **my-virtualenv-name _kernel_** in jupyter.
