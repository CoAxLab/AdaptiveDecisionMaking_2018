# AdaptiveDecisionMaking_2018 (ADM)
Repository for code and lab resources for "Neural and Cognitive Models of Adaptive Decision Making" course (2018)


## Jupyter notebooks
Click on [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/CoAxLab/AdaptiveDecisionMaking_2018/master) to run jupyter notebooks for labs and homework.


## Instructions for getting started

- [**Anaconda distribution**](https://www.anaconda.com/): package management for scientific python (& R)

  - [**Jupyter**](http://jupyter.org/): interactive python interpreter in your browser ([tutorial](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46))
  - [**pandas**](http://pandas.pydata.org/pandas-docs/stable/): tabular dataframe manager ([tutorial](https://medium.com/init27-labs/intro-to-pandas-and-numpy-532a2d5293c8))
  - [**numpy**](http://www.numpy.org/): numerical computing library ([tutorial](https://www.machinelearningplus.com/python/101-numpy-exercises-python/))
  - [**scikit-learn**](http://scikit-learn.org/stable/): data science and machine learning library ([tutorial](http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/text_analytics/general_concepts.html))
  - [**matplotlib**](https://matplotlib.org/index.html): plotting and visualization library ([tutorial](https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python))
  - [**seaborn**](https://seaborn.pydata.org/): wrapper for making matplotlib pretty, plays nice w/ pandas ([tutorial](https://elitedatascience.com/python-seaborn-tutorial))
  - [**and more...** ](https://docs.anaconda.com/anaconda/packages/pkg-docs/)

- Install **Anaconda** with **Python 3.6**:

  - [**OSX**](https://www.anaconda.com/download/#macos)
  - [**Linux**](https://www.anaconda.com/download/#linux)
  - [**Windows**](https://www.anaconda.com/download/#windows)



#### Confirm Anaconda and Python installed

```bash
# check that your system is now using anaconda's version of python
$ which python
# output: /Users/kyle/anaconda3/bin/python

# and that you installed Python 3.6
$ python -V
# output: Python 3.6.0 :: Anaconda custom (64-bit)
```



#### Install ADMCode package

[**ADMCode**](https://pypi.org/project/ADMCode/) is a python package with custom code that can be used to complete the labs and homeworks (which will both be in the form of jupyter notebooks)

```bash
$ pip install --upgrade ADMCode
```



#### Run Jupyter Notebook locally

```sh
# open up a terminal and execute the command below
$ jupyter notebook
```

- Useful walkthrough of jupyter notebook features ([here](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46))
