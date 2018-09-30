# AdaptiveDecisionMaking_2018 (ADM)
Repository for code and lab resources for "Neural and Cognitive Models of Adaptive Decision Making" course (2018)


### Jupyter notebooks [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/CoAxLab/AdaptiveDecisionMaking_2018/master)
Click on binder badge above to run jupyter notebooks for labs and homework. Or download the ipynb files [**here**](https://nbviewer.jupyter.org/github/CoAxLab/AdaptiveDecisionMaking_2018/tree/master/notebooks/) to run locally.


## Instructions for getting started
#### Install **Anaconda** with **Python 3.6**:
  - [**OSX**](https://www.anaconda.com/download/#macos)
  - [**Linux**](https://www.anaconda.com/download/#linux)
  - [**Windows**](https://www.anaconda.com/download/#windows)

#### Confirm installs
```bash
# check that your system is now using Anaconda's python
which python
```
```bash
# and that you installed Python 3.6
python -V
```



## Install ADMCode package
[**ADMCode**](https://pypi.org/project/ADMCode/) is a python package with custom code that can be used to complete the labs and homeworks (which will both be in the form of jupyter notebooks)
```bash
pip install --upgrade ADMCode
```



## Working with `git`
Git is full of weird nonsense terminology. [**This tutorial**](http://rogerdudler.github.io/git-guide/) is a super useful resource for understanding how to use it.

- If you don't already have a github account, create one [**here**](https://github.com)
- Install git command-line tools (see *setup* section [**here**](http://rogerdudler.github.io/git-guide/))

#### Clone ADMCode
* Open a terminal and `cd` to a directory where you want to download the ADMCode repo (example: `cd ~/Dropbox/Git/`)
* Next, use `git` to `clone` the *remote* ADMCode repository to create a *local* repo on your machine
```bash
# make sure you've done steps 1 and 2
# before executing this in your terminal
git clone https://github.com/CoAxLab/AdaptiveDecisionMaking_2018.git
```

#### Pull updates
* Use `git pull` to update your local repo with any changes to the *remote* ADMCode repo
* In the command below, `origin` is the default name pointing to the remote repo on Github
* `master` is the `branch` of the remote that you want to sync with
```bash
# first cd into your local ADMCode repo
# (same directory as step 1 in "Clone ADMCode" ^^^)
git pull origin master
```

## Useful resources
- [**Anaconda distribution**](https://www.anaconda.com/): package management for scientific python (& R)
- [**Jupyter**](http://jupyter.org/): interactive python interpreter in your browser ([tutorial](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46))
- [**pandas**](http://pandas.pydata.org/pandas-docs/stable/): tabular dataframe manager ([tutorial](https://medium.com/init27-labs/intro-to-pandas-and-numpy-532a2d5293c8))
- [**numpy**](http://www.numpy.org/): numerical computing library ([tutorial](https://www.machinelearningplus.com/python/101-numpy-exercises-python/))
- [**scikit-learn**](http://scikit-learn.org/stable/): data science and machine learning library ([tutorial](http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/tutorial/text_analytics/general_concepts.html))
- [**matplotlib**](https://matplotlib.org/index.html): plotting and visualization library ([tutorial](https://www.datacamp.com/community/tutorials/matplotlib-tutorial-python))
- [**seaborn**](https://seaborn.pydata.org/): wrapper for making matplotlib pretty, plays nice w/ pandas ([tutorial](https://elitedatascience.com/python-seaborn-tutorial))
- [**and more...** ](https://docs.anaconda.com/anaconda/packages/pkg-docs/)
