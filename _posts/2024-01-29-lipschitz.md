---
layout: post
title: What the Lipschitz?!
date: 2024-01-29 11:12:00-0400
description: A short and easy to follow primer on convexity.
tags: math optimisation
categories: mathematics
related_posts: false
---

## Introduction

Lipschitzness of a function is essential for ensuring the convergence properties of many gradient descent algorithms

## Definition

Simply put, Lipschitz functions are those which do not explode for some value $$x$$. So, functions which change too fast and/or become infinitely steep are not Lipschitz functions.

More formally, let $$\chi \in \mathbb{R}^{d}$$ be a d-dimensional subspace of real values. If we take a function $$f: \mathbb{R}^{d} \rightarrow \mathbb{R}^{n}$$ which provides a mapping from a d-dimensional space to a p-dimensional space, we can say that $$f$$ is $$L$$-Lipschitz over $$\chi$$ if and only if we have:

$$
|f(x_{2}) - f(x_{1})| \le L | x_{1} - x_{2} | \qquad \forall x_{1}, x_{2} \in \chi
$$

I always think this looks a bit confusing written this way, so a more intuitive way to write it is simply:

$$
\frac{|f(x_{2}) - f(x_{1})|}{ | x_{1} - x_{2} | } \le L \qquad \forall x_{1}, x_{2} \in \chi
$$

where we now have the form of $$\Delta y / \Delta x$$ on the left hand side.

Thinking about this a bit more, this condition demands that the slope of the secant line between two points $$x_{1}$$ and $$x_{2}$$ must be between $$-L \le m \le L$$.

### Example: Is cosine Lipschitz?

Start by employing the definition above,

$$
\frac{f(x) - f(y)}{x - y} \simeq f^{'}(x) = \text{sin}(x)
$$

We know that $$|\text{sin}(x)| \le 1$$, so we can rewrite this as:

$$
|f(x) - f(y)| \le L | x - y |
$$

where $$L = 1$$. So we say that cosine is a $$1$$-Lipschitz function.

Now, if you're screaming "Stop showing me maths!!", you're in luck, because I've created some nice plots for us to look at...

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/blogs/lipschitz/lipschitz_curves.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Plots showing linear, cosine, cusp and quadratic functions to showcase Lipschitzness.
</div>


## Globally & Locally Lipschitz

Just to round off this small post, I want to talk a bit about global and local Lipschitz functions. The above section kind of describes functions which are globally Lipschitz since we haven't defined a subset of the function space to consider, so here, we'll start local!

Say for example, we have some function $$f$$ which is locally Lipschitz for a compact subset of $$\chi$$. We'll call this $$\Omega \subset \chi \in \mathbb{R}^{d}$$.

For the local Lipschitz property to hold, it must be true that there is a constant $$L_{\Omega}$$ such that,

$$
|f(x) - f(y)| \le L_{\Omega} | x - y | \qquad \forall x_{1}, x_{2} \in \Omega
$$

where $$L_{\Omega}$$ can indeed depend on the subset $$\Omega$$. For example, the function $$f(x) = x^{2}$$ has an $$L_{\Omega}$$ which depends on $$x$$ (since only one of the $$x^{2}$$ will cancel). Therefore as the subset $$\Omega$$ becomes larger, $$L_{\Omega}$$ will scale linearly with this.

The above example defines a situation where we have local Lipschitzness but not global!

For global Lipschitzness, we require the function to have a Lipschitz constant which does not depend on the subset $$\Omega$$, i.e. $$L_{\Omega} = L$$.

I hope this post has been a useful primer into the property of Lipschitzness. As always, feel free to reach out with any comments/questions!



<!-- Local Lipschitz continuity means that a function is Lipschitz continuous on every compact subset of its domain. Let's discuss the local Lipschitzness of the functions we plotted:

Linear Function: It is globally Lipschitz continuous, so it is also locally Lipschitz continuous everywhere.

Sine Function: The sine function is globally Lipschitz continuous with a Lipschitz constant L = 1 (since its derivative is bounded by 1 in absolute value), so it is also locally Lipschitz continuous everywhere.

Cusp Function: The cusp function (f(x) = |x|) is not globally Lipschitz continuous because the derivative is not bounded near the cusp (x = 0). However, it is locally Lipschitz continuous away from the cusp. For any compact subset that does not include the origin, we can find a Lipschitz constant.

Quadratic Function: The quadratic function (f(x) = x^2) is not globally Lipschitz continuous because its derivative grows without bound as x increases. However, it is locally Lipschitz continuous on any compact subset of its domain because within any bounded interval, the derivative of the function is bounded, and thus a Lipschitz constant exists. -->