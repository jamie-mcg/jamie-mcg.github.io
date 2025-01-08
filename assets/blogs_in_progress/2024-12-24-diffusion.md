---
layout: post
title: Diffusion
date: 2024-01-29 11:12:00-0400
description: A comprehensive look at the theory and concepts underpinning diffusion models.
tags: math diffusion
categories: "deep learning"
related_posts: false
featured: true
---

Generative Models are a class of networks which have left many of us spell-bounded in recent years. These models have propelled Artificial Intelligence (AI) into popular culture and captivated the general public by their attempts to encapsulate something inately human; _creativity_.

In parallel to many intricate "under-the-hood" technical advancements, the humanisation of AI has played a major role in driving the popularity of generative models. The ease of use and access to some of the most powerful AI tools the world has seen, has motivated a widespread adoption of generative models into many services and third-party applications.

Indeed, as Uncle Ben would rightly say:

_with great power, comes great responsibility_

and AI is no exception, with many of these new possibilities bringing with them contraversies and a new requirement for AI governance.

Anyway, we can cover AI safety and governance in another blog post. For now, in this post, I want to explain how diffusion models really work and discuss some of their incredible applications.

## Problem Setup

As with all Machine Learning problems, we must start with the data! The goal of generative modelling solutions is to find some model for the data distribution such that generated outputs lie within the distribution of inputs.

Let's just contextualise that with an example: Imagine I asked you to imagine a car. It is highly likely that the image you choose to imagine would be similar to those you have seen in the past. In this case, your brain has a model of what a car should look like, that has been learnt from all your experience of seeing cars in the past. Therefore we can say that, the imagined output has a high probability of being close to one which is realistic to find in the wild.

To be clear, this doesn't mean that we can imagine exactly a particular image of a car and reproduce it in our head (unless you are a slightly obsessive petrolhead...), it means we recognise the key concepts that are required for a car to be, well... a car, and we are able to compose these concepts together into something that resembles a car. Indeed, our model is not discrete and is also able to interpolate between many concepts to produce something entirely different to what we may have seen in the past. Without sounding too philisophical, this is arguably one of the pillars of human-like creativity, the compositionality of learned concepts.

Now let's just round off this section by stating some mathematical definitions which will be useful later.

Let $x \in RR^d$ be our data with dimension $d$ which is obtained from some underlying distribution $q_{\text{data}}(x)$. The goal of a generative model is to estimate $p(x)$ which approximates the true data distribution $q_{\text{data}}(x)$.

So, how do we do this?

## Setting up the training algorithm

We have data, but we don't have any labels, so at the moment we're in an unsupervised regime. Typically learning from unsupervised data tends to be quite hard and is also somewhat hard to control - since you are leaving the model up to its own devices.

A much easier problem would be a supervised task, so let's try and set one of these up from the ingredients we have.

In much the same way as you would start from a blank page when asked to draw a picture, we can ask our probabilistic model to do the same. In this setting, we have the input as a randomised vector of the same dimension of $x$ and $x$ becomes our target! Voila!

Right, before we get confused, lets add to our math we had earlier, just so we can digest this.

Let our data be $x^{(0)}$ and our starting point (blank page/random vector) be $x^{(T)}$. Apologies for the cryptic notation, I promise this notation will become clearer later...

The problem we have set up is $x^{(0)} \sim p(x^{(0)}|x^{(T)})$ as an approximation to $x \sim q(x)$.

Hmm... this still seems super hard though, right? If I asked you to go from a blank page to a masterpiece in a single step, unless you were Picasso, you would quite clearly struggle. So let's break it up, step-by-step.

Instead of going from $x^{(T)} \rightarrow x^{(0)}$ in 1 step, lets say it takes $T$ steps:

$$
x^{(0)} \leftarrow x^{(1)} \leftarrow \dots \leftarrow x^{(t-1)} \leftarrow x^{(t)} \leftarrow \dots \leftarrow x^{(T-1)} \leftarrow x^{(T)}
$$

where now we can definea much simpler regression problem at each step, which can be modelled by $x^{(t-1)} \sim p(x^{(t-1)}|x^{(t)})$. The result being a 
