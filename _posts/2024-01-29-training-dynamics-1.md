## Introduction


## Setting up our playground

Althogh simple, linear regression is a powerful setting for us to gain intuition around more complex neural network phenomena. Indeed, with this extremely simple playground, many questions surrounding the dynamics of optimization can be answered analytically.

Let's begin with a simple linear regression model that has a mapping $R^{N}\rightarrow R$ (i.e. the output is a scalar value):

$$
y = \mathbf{w}^{T}\mathbf{\psi}(\mathbf{x}) + b
$$

where $\mathbf{\psi}(\mathbf{x}) \in R^{N}$ is the feature representation of the input vector $\mathbf{x} \in R^{N}$ and $\mathbf{w} \in R^{N}$ is a weight vector that, together with the bias $b \in R$, defines the parameters of our toy model.

Next, every great model needs a problem to solve. So let's take our loss function to be a regression loss of:

$$
\mathcal{L} = \frac{1}{2}||\mathbf{w}^{T}\mathbf{\psi}(\mathbf{x}) + b - \hat{y}||^{2}
$$

where $\hat{y} \in R$ is the true value we are attempting to predict for some input $\mathbf{x}$.

Given some data pairs $\mathcal{D} \sim \{\mathbf{x}_{i}, y_{i}\}^{N}_{i=1}$, we now wish to minimise the function $\mathcal{L}$ above with

$$
w^{*} \in argmin_{w} \mathcal{L}(\mathbf{w}) 
$$
$$
= argmin_{w} \frac{1}{2N} \sum_{i=1}^{N} (\mathbf{w}^{T}\mathbf{\psi(\mathbf{x}_{i})} + b - \hat{y}_i)^{2}
$$
$$
= argmin_{w} \frac{1}{2N} ||\mathbf{W}^{T}\mathbf{\Psi(\mathbf{x}_{i})} - \hat{Y}||^{2}
$$

where in the above, we have absorbed the bias into the weight vector $\mathbf{W} = (\mathbf{w}, b)$ and defined $\mathbf{\Psi} = (\mathbf{\psi}, 1)$.

Note that in the 1st line of the argmin equation above, we also use the symbol $\in$ as the minimum is by no means guaranteed to be unique - in fact it is much more likely in general to be part of a set. In these cases, the optimum that we find during training will depend entirely upon the dynamics.

Let's take a quick look at what we have so far though. In particular, the function we really want to understand is the loss function!

Note that in the above we can see this loss function is indeed a convex quadratic, meaning that our dynamics should be very well behaved (i.e. positive curvature everywhere and a unique global minimum).

Aside: Everyone always throws the term convex about in the literature as a way to explain away a lot of training dynamics. When you're starting out, it is easy to overlook what this term really means! Take a look at my other post on convexity to give yourself a brief primer on this subject!



