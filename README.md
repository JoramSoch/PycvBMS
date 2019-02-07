# PycvBMS

This module collects methods for calculating the cross-validated log model evidence (cvLME).

Currently, it only features the general linear model (GLM); more functionality will come soon.

<h3> Getting Started </h3>

```python
import cvBMS
import numpy as np

# have your data ready
Y   # an n x v  data matrix
X1  # an n x p1 design matrix from one model
X2  # an n x p2 design matrix from another model
V   # an n x n  covariance matrix, probably V = np.eye(n)

# two models
m1 = cvBMS.GLM(Y, X1, V)
m2 = cvBMS.GLM(Y, X2, V)

# model space
ms = cvBMS.MS(np.r_[m1.cvLME(), m2.cvLME()])
PP = ms.PP()    # posterior model probabilities
```
