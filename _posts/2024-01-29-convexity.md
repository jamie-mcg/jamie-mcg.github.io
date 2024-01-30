---
layout: post
title: a post with math
date: 2024-01-29 11:12:00-0400
description: A short and easy to follow primer on convexity.
tags: formatting math
categories: mathematics
related_posts: false
---

## Introduction

## What is convexity?

Quote:
A function is convex if the line segment between two points lies above the function.

FIGURE

In the above figure, we see an illustrative example of the definition above. Here, we can see that if the intersection of the line forms an enclosed area which is uninterrupted above the curve, the function can be considered convex.

We can also see that if the opposite is true i.e. the enclosed area is below the curve, the function is concave!

Indeed, a convex function $$f(x)$$ can be reflected into a concave function $$-f(x)$$.

Mathematically, convexity is best described by Jensen's inequality,

EQUATION

JENSEN CHAT

Additionally, for a twice differentiable function which is convex, we have two rather nice conditions:

$$
f^{''}(x) >= 0 \forall x,
f(y) >= f(x) + \nabla f(x)^{T} (y-x).
$$

The first of these conditions tells us that curvature must be positive everywhere. The second condition ensures that the line between two points $x$ and $y$ lies above the function within that interval (illustrated earlier).

Following on from this, the first condition ensures that a convex function must curve upwards (or not at all). Therefore, if we find a minima x, any movement away from this point will result in an increase in the function value.

Together, these conditions are enough to guarantee that any minima found in a convex function must be a global minima (although this doesn't necessarily need to be unique). This is extremely useful for techniques such as stochastic Gradient Descent since if we find a local minima (something which Gradient Descent generally tends to find), we can be certain that this in fact the global minima - the best possible solution.

## The Strong, the Strict and the Standard

As always with math, we don't just have one type of something, we have some vague terms that sound cool to describe some more features. 

Convex functions are no exception and can be further categorised with three properties:

- Convex
- Strictly Convex
- Strongly Convex

Now, the further you move down that list, the stronger these properties become (hence the "strongly" term). This just means that the subset of convex functions that have these properties becomes smaller due to stronger constraints.

So far in this post, we have been talking about convexity in its most general form, where $$f^{''}(x) >= 0$$. This is the condition for 'convex' functions.

Strictly convex functions are those which satisfy $$f^{''}(x) > 0$$, i.e. the curvature can never be 0.

Stongly convex functions are those which satisfy $$f^{''}(x) >= m > 0$$, where the curvature is non-vanishing and stays bounded below by some positive value $m$.

