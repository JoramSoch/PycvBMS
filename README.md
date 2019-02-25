# PycvBMS

<b>Note: This repository is now obsolete. Please consider using the <a href="https://github.com/JoramSoch/cvLME">cvLME package</a>!</b>

<br>

<h3>Python module for cross-validated Bayesian model selection</h3>

This module collects methods for calculating the cross-validated log model evidence (cvLME).

It features the general linear model (GLM) and the Poisson distribution with exposures (Poiss).

Below are two code snippets that help you getting started when estimating a GLM or a Poisson.

<br>

<h3> Getting started with the GLM </h3>

Here, posterior probabilities of two GLMs with arbitrary design matrices (X1, X2) are calculated.

```python
import cvBMS
import numpy as np

# have your data ready
Y   # an n x v  data matrix of measured signals
X1  # an n x p1 design matrix from one model
X2  # an n x p2 design matrix from another model
V   # an n x n  covariance matrix, probably V = np.eye(n)

# two models
m1 = cvBMS.GLM(Y, X1, V)
m2 = cvBMS.GLM(Y, X2, V)

# model space
ms = cvBMS.MS(np.r_[[m1.cvLME()], [m2.cvLME()]])
PP = ms.PP()    # posterior model probabilities
```

<br>

<h3> Getting started with the Poisson </h3>

Here, log Bayes factors of a Poisson with exposures (m1) vs. a Poisson without exposures (m0) are calculated.

```python
import cvBMS
import numpy as np

# have your data ready
Y   # an n x v data matrix of measured counts
x   # an n x 1 design vector of exposure values

# two models
m0 = cvBMS.GLM(Y)
m1 = cvBMS.GLM(Y, x)

# model space
ms  = cvBMS.MS(np.r_[[m0.cvLME()], [m1.cvLME()]])
LBF = ms.LBF()      # log Bayes factors
```
