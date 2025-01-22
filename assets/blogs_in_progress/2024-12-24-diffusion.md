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

Let's just contextualise that with an example: say I asked you to imagine a car. It is highly likely that the image you choose to imagine would be similar to those you have seen in the past. In this case, your brain has a model of what a car should look like, that has been learnt from all your experience of seeing cars in the past. Therefore we can say that, the imagined output has a high probability of being close to one which is realistic to find in the wild.

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

where now we can define a much simpler regression problem at each step, which can be modelled by $x^{(t-1)} \sim p(x^{(t-1)}|x^{(t)})$. With this, we can write the marginal distribution as:

$$
p(x^{(0)}) = \int p(x^{(T)}) p(x^{(0)}|x^{(1)}) \dots p(x^{(T-1)}|x^{(T)}) dx^{(1)} \dots dx^{(0)} \\
= \int p(x^{(T)}) \prod_{t=1}^{T} p(x^{(t-1)}|x^{(t)}) dx^{(t)} \\
= \int p(x^{(0:T)}) dx^{(1:T)}
$$

where the last line is a commonly used shorthand for the concatenation of variables across timesteps.

OK, so the math checks out that as long as we obey the time ordering i.e. $x^{(T)}$ being the worst, $x^{(0)}$ being the best and $t=T-1,\dots 1$ being progressively better versions of $x$.

## Creating $x^{(t)}$

The first problem we have is that $x^{(0)} \sim q(x^{(0)})$ is a perfect data sample, so how do we go about getting our hands on $x^{(t)} \forall t \in \{1, \dots, T\}$.

This is where "noising" comes in!

The data creation process for our litter regression problems will involve noising our perfect data $x^{(0)}$ by a small amount over $T$ steps, until we reach $x^{(T)}$.

We will denote the noising process as $q(x^{(t)}|x^{(t-1)})$ and define it as a first order Gaussian which performs the transformation:

$$
x^{(t)} = \lambda_t x^{(t-1)} + \sigma_t \epsilon_t \qquad \epsilon \sim \mathcal{N}(0,1) \\
q(x^{(t)}|x^{(t-1)}) = \mathcal{N}(x^{(t)}; \lambda_t x^{(t-1)}, \sigma_t^2)
$$

Great! We now have our noising and denoising processing written down!


## How do we noise this thing?

In order to actually use this noising process, we need to first define $\lambda_t$ and $\sigma_t$. If we stop and think about what these parameters actually mean for a second, we might be able to gain some insight.


- $\lambda_t \rightarrow 0$ means that we forget all information and only have noise
- $\lambda_t = 1$ means we retain the information in $x^{(t-1)}$

We want $x^{(T)}$ to contain zero information, since $x^{(T)}$ is our blank canvas which we want to easily sample from whatever we want to generate something new.

In this sense, we would like $\lambda_t < 1$, but we want to keep small regression tasks as simple as possible - not lose too much information at each step. So we might want to have $\lambda_t$ somewhere close to 1 in practice.

As long as $\lambda_t < 1$, for $T \rightarrow \infty$, we are guaranteed to get $x^{(T)}$ as pure noise! - a perfect blank canvas for an ML model.

Similarly for $\sigma_t$, this controls the amount of noise that is added at each step. Something we might want to put $\sim 0$ for the same reasons as above.

A common choice is to set $\sigma_t^2 = 1 - \lambda_t^2$, which for now we will motivate by saying that when $\lambda_t \ge 1$, $\sigma_t^2 \ge 0$. But it is actually theoretically justified by guaranteeing that $\mathbb{E}_{q(x^{(t)})}[x^{(t)}] = 0$ and $\mathbb{E}_{q(x^{(t)})}[(x^{(t)})^2] = 1$.

### Variance Preserving Process

Aside: $\sigma_t^2 = 1 - \lambda^2_t$ ensures that $\mathbb{E}_{q(x^{(t)})}[x^{(t)}] = 0$ and $\mathbb{E}_{q(x^{(t)})}[(x^{(t)})^2] = 1$.

We know that the data satisfies:

$$
\mathbb{E}_{q(x^{(0)})}[x^{(0)}] = 0 \quad \text{and} \quad \mathbb{E}_{q(x^{(0)})}[(x^{(0)})^2] = 1
$$

because it can be normalised to do so. So let's consider $x^{(1)}$:

$$
\mathbb{E}_{q(x^{(1)})}[x^{(1)}] = \mathbb{E}[\lambda_1 x^{(0)} + \sigma_1 \epsilon_1] = \lambda_1 \mathbb{E}[x^{(0)}] + \sigma_1 \mathbb{E}[\epsilon_1] = 0
$$

because $\epsilon_1 \sim \mathcal{N}(0, 1)$. And also:

$$
\mathbb{E}_{q(x^{(1)})}[(x^{(1)})^2] = \mathbb{E}[\lambda_1^2 (x^{(0)})^2 + \sigma_1^2 \epsilon_1^2] = \lambda_1^2 \mathbb{E}[(x^{(0)})^2] + \sigma_1^2 \mathbb{E}[\epsilon_1^2] = \lambda_1^2 + \sigma_1^2 \\ \therefore \mathbb{E}_{q(x^{(1)})}[(x^{(1)})^2] = 1 \iff \sigma_1^2 = 1-\lambda_1^2
$$

By recursion, we can see that this is true for all $t$.

In this case, when the process is variance preserving, we can write down:

$$
q(x^{(t)}|x^{(0)}) = \mathcal{N}(x^{(t)}; \prod_{i=1}^{t} \lambda_{i} x^{(0)}, 1 - \prod_{i=1}^{t}\lambda_t^2)
$$

which can be proved by unrolling $q(x^{(t)}|x^{(t-1)})$, if you fancy doing some rather laborious maths.

OK, now we are getting somewhere. We have a problem setup, we have our chain of makeshift supervised regression problems and we know how to generate the noisey labels for each of these regression problems. Indeed, from the above, we can even generate any noisey sample at any point in this chain, directly from the input $x^{(0)}$, as long as we have a variance preserving process!

Practically, this means that we do not need to generate _all_ the previous $[x^{(1)}, \dots x^{(t-1)}]$ just to get $x^{(t)}$, which will become important later when we talk about training!

Just one thing to note here before we move on to the next section is that; in order for us to generate $x^{(0)}$ from $x^{(T)}$, which is essentially the aim of a diffusion model, we still require $x^{(T)}$ to be an _easy_ sample to generate. To put this another way, when we deploy this model we will not have access to $x^{(0)}$, we need to generate it! So our starting point $x^{(T)}$ should ideally not have any information about $x^{(0)}$ left inside it, it should be complete noise - which is super easy to generate independent of any other sample in the chain.

The above requirement basically tells us that we're going to run into a balancing act between how _hard_ we make each regression problem (how much noise we add per-step) and how many timesteps can afford to do. Anyway, more on this later!

## Defining a Loss

*Figure*

Taking a step back, we have a dataset that now consists of lots of ordered noisey samples which need to be assigined to a particular regression step in our denoising schedule (i.e., going from $T \rightarrow$ 0).


To play aroung a bit, lets simply define a loss which concatenates all of our small regression tasks, each with their own set of parameters $\Theta = \{\theta_t\}_{t\in [0, T-1]}$.

$$
\mathcal{L}(\Theta) = \mathbb{E}_q \log\left\{p(x^{(T)}) \prod_{t=1}^{T} p_{\theta_{t-1}}(x^{(t-1)}|x^{(t)}) \right\}
\\
=\mathbb{E}_q \left[\log p(x^{(T)})\right] + \sum_{t=1}^{T} \mathbb{E}_q \left[\log p_{\theta_{t-1}}(x^{(t-1)}|x^{(t)})\right]
$$

where it is the second term which depends on the parameters, so an individual regression loss would be $\mathcal{L}(\theta_{t-1}) = \mathbb{E}_q \left[\log p_{\theta_{t-1}}(x^{(t-1)}|x^{(t)})\right]$.

OK cool, so as we are all excellent researchers and like to think about the implications of our maths before whacking it into some code... Let's just have a think about the complexity of the above formalism.

Let $\theta_t \in \mathbb{R}^d$, then we will end up with a total parameter count of $T \times d$. OK, so this means that as the number of timesteps increases, this model is going to scale linearly and as $T \rightarrow \infty$, become intractable.

OK, you're right, we'll never be actually taking $T \rightarrow \infty$ in practice. But we do need it to be pretty large to satisfy the requirements we discussed earlier, so its still a problem. One way we can try to solve this problem is _weight sharing_.

From now on, we will define one set of weights for all timesteps $\theta_t \rightarrow \theta$, which reduces the total parameters to just $d$. This is not only motivated from a computational perspective, but also by the fact that if we have defined our chain of regression tasks correctly, they are all kind of doing the same thing. So intuitively, there should be some redundancy across timesteps.

After all this, we can make a minimal adjustment to our individual loss function:

$$
\mathcal{L}_{t-1}(\theta) = \mathbb{E}_q \left[\log p_{\theta}(x^{(t-1)}|x^{(t)})\right].
$$

Right, last thing for this section, how do we add some control to this training? Let me explain...

Simply summing all the individual losses for each step in our chain is fine, but it weights each step equally in terms of its feedback to $\theta$ - which are now shared across timesteps. However, you can imagine that there might be some steps that need to be _more accurate_ than others in our chain.

In order to add some contol, we will introduce some weighting parameters $w_t$ into our total loss which control how much each step contributes to updating $\theta$:

$$
\theta^{*} = \text{argmin}_{\theta} \sum_{t=1}^{T} w_{t-1} \mathcal{L}_{t-1}(\theta)
$$

## The Magic of Stochastic Optimization

Right now, you're probably thinking, how much more can there be to this! But honestly, this is one of my favourite tricks that is used all over Machine Learning. Its super simple but when I first started learning about diffusion models, understanding how they leverage stochastic optimization made me think about the time axis (and also the sequence axis in transformers) in a much clearer way.

Anyway, the above update rule has a sum over $T$ terms, all of which require varying numbers of passes through the network. This is fine but because we are sharing weights across timesteps, we can do better.

To gain an intuition, lets go back to thinking about batches of data and training a very simple neural network with parameters $\theta$. Imagine our dataset $\mathcal{D}$ contains $N_\mathcal{D}$ samples. Theoretically, this gives us a solution of the form:

$$
\theta^{*} = \text{argmin}_{\theta} \frac{1}{N} \sum_{n=1}^{N} w_{n} \mathcal{L}_{n}(\theta)
$$

where $w_n$ are weighting scalars for how much each sample contributes - this could be related to the quality of that datapoint for example.

This looks very similar to the equation we just wrote down for diffusion models, right? But in training a neural network we don't need see _all_ the $N_\mathcal{D}$ samples, we usually get away with defining a some batches and approximating the weighted sum via an expectation over the data:

$$
\frac{1}{N} \sum_{n=1}^{N} w_{n} \mathcal{L}_{n}(\theta) = \mathbb{E}_{n \sim \mathcal{U}(1,N)}[w_n \mathcal{L}_{n}(\theta)]
$$

This trick is possible because the parameters are shared across mini-batches. But remember, in our diffusion setup, we are also sharing parameters across $T$ timesteps, so we can employ exactly the same trick!

$$
\theta^{*} = \text{argmin}_{\theta} \sum_{t=1}^{T} w_{t-1} \mathcal{L}_{t-1}(\theta) = \text{argmin}_{\theta} \left\{\mathbb{E}_{t \sim \mathcal{U}(1, T)} \left[w_{t-1} \mathcal{L}_{t-1}(\theta)\right]\right\}
$$

Hooray, now we don't need ALL $T$ terms! We can just sample timesteps randomly from a uniform distribution. But remember, because we have a variance preserving process, we can get the target for the sampled timestep directly from the data $x^{(0)}$.

## The Denoising Step

Now we come to the denoising part of our pipeline... Finally!

In this part, the goal is to learn the noise which we want to remove at each step. However, because we have already defined the noising schedule with a Gaussian we can also parameterise the noise to remove similarly:

$$
p(x^{(t-1)}|x^{(t)}; t-1, \theta) = \mathcal{N}(x^{(t-1)};\mu_{\theta}(x^{(t)}; t-1), \sigma_{\theta}^{2}(x^{(t)}; t-1))
$$

This leaves us with two terms to deal with here; the mean and the variance.

For the mean, we can do this by either; (i) directly parameterising $x^{(0)}$ at each step, or (ii) by parameterising the noise directly $\epsilon^{(t)}$.

### i. $x^{(0)}$-parameterisation

This method involves looking at the noising step $q(x^{(t-1)}|x^{(0)}, x^{(t)})$ where from Bayes rule, we have:

$$
q(x^{(t-1)}|x^{(0)}, x^{(t)}) \propto q(x^{(t-1)}|x^{(0)})q(x^{(t)}|x^{(t-1)})  \\
= \mathcal{N}\left( x^{(t-1)};\ \left(\prod_{t^\prime = 1}^{t-1} \lambda_{t^\prime} x^{(0)} \right),\ 1 - \prod_{t^\prime = 1}^{t-1} \lambda_{t^{\prime}}^{2} \right) \times \mathcal{N}(x^{(t)};\ \lambda_t x^{(t-1)}, \ 1-\lambda_t^2)
$$

then completing the square and doing some mathematical gymnastics gives us a new distribution $q(x^{(t-1)}|x^{(0)}, x^{(t)}) = \mathcal{N}(x^{(t-1)}; \mu_{t-1|0,t}, \sigma_{t-1|0,t}^{2})$ where:

$$
\mu_{t-1|0,t} = \frac{\left(\prod_{t^\prime = 1}^{t-1} \lambda_{t^\prime}\right)(1-\lambda_t^2)}{1 - \prod_{t^\prime = 1}^{t} \lambda^2_{t^\prime}} x^{(0)} + \frac{\left(1 - \prod_{t^\prime = 1}^{t-1} \lambda^2_{t^\prime}\right)\lambda_t}{1 - \prod_{t^\prime = 1}^{t} \lambda^2_{t^\prime}}x^{(t)} \\
\sigma_{t-1|0,t}^{2} = \frac{\left(1 - \prod_{t^\prime = 1}^{t-1} \lambda^2_{t^\prime}\right)(1 - \lambda_t^2)}{1 - \prod_{t^\prime = 1}^{t} \lambda^2_{t^\prime}}
$$

In the $x^{(0)}$-parameterisation, we do exactly what it says on the tin and parameterise $x^{(0)}$:

$$
\mu_{t-1|0,t}^{\theta} = a^{(t)}x^{(0)}_{\theta}(x^{(t)}) + b^{(t)}x^{(t)}
$$

### ii. $\epsilon$-parameterisation

An alternative parameterisation to the above is found by instead aiming to predict the noise $\epsilon$ at each step. Similarly to the $x^{(0)}$-parameterisation, our weight sharing is well-founded by the argument that each step is adding a similar amount of noise - so we can imagine some redundancy in the parameters across adjacent steps.

If we cast our minds back to earlier, we had a noising schedule which can be written as:

$$
x^{(t)} = \prod_{t^\prime = 1}^{t} \lambda_{t^\prime} x^{(0)} + \sqrt{1-\prod_{t^\prime = 1}^{t}\lambda_{t^\prime}^{2}}\ \epsilon^{(t)} \qquad \epsilon^{(t)} \sim \mathcal{N}(0,1) \\
x^{(t)} = c^{(t)} x^{(0)} + d^{(t)} \epsilon^{(t)}
$$

So if we take the conditional mean that we got from our $x^{(0)}$-parameterisation, we can easily substitue the above into this and write down a new expression of the form:

$$
\mu_{t-1|0,t}^{\theta} = \left(\frac{a^{(t)}}{c^{(t)}} + b^{(t)}\right)x^{(t)} - \frac{a^{(t)}d^{(t)}}{c^{(t)}}\epsilon^{(t)}_{\theta}(x^{(t)})
$$

So given the input and the timestep, our parameterisation now predicts the noise that needs to be taken away from $x^{(t)}$ in order to give $x^{(t-1)}$.

OK, so hopefully its pretty easy to see how these two parameterisations are related to each other. In fact, you might be asking, _"what is the difference?"_ or _"why would we want to do this?"_ - these would be valid questions!

To answer these a bit more in depth, we can consider how each one is actually performing the denoising step and at the same time, learn about some related work!

## Denoising Autoencoders

