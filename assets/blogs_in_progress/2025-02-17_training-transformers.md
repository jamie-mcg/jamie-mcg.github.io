---
layout: post
title: Parallel Training the Transformer Architecture
date: 2024-01-29 11:12:00-0400
description: Schemes for parallelism of LLMs during training.
tags: LLM engineering
categories: "deep learning"
related_posts: false
featured: true
---

In order to be able to train the worlds largest machine learning models, many engineering techniques and strategies must be employed. But before we get onto those, lets just set the scene a little!

When you train your first ever neural network using a GPU, chances are that you will be doing this on a single chip. This is great but at some point, this chip will be bottlenecked by its memory bandwidth and/or the number of FLOPs it can execute. What do we mean by this? Well... every GPU or TPU will have an on-chip memory wherby information is temporarily stored before being transferred to the computational elements in the processor, the rate at which this memory transfer can happen is determined by the _memory bandwidth_ and the rate at which the processor can process the information is determined by the _FLOPs_ (Floating Point Operations). So, in order to scale models efficiently, overcome fundamental constraints such as VMEM limits, and keep training to within a reasonable timescale, it becomes necessary to introduce multiple chips into a cluster.

As you can imagine, this increase in compute gives us a few more options in how to arrange our training pipeline. But also some more potential bottlenecks to consider which are related to balancing the utilisation of all chips (how much computation they are being used for at any one time) with the inter-chip communication latency -- in particular, we will see how we can attempt to overlap these two in order to ensure as high utilisation as possible during training.

## Aside: MLPs Are All You Need

OK, clearly this section heading is misleading. But let me explain! So, transformer blocks are essentially made up of an attention part and a feedforward part. This feedforward block is further composed of three dense layers as below:


In maths, for a model with $d_{\text{model}}$ embedded/model dimension and $d_{\text{hidden}}$ hidden dimension, and an input with batch size $N$ and sequence length $T$, this can be written as:

$$
X_{\text{out,ff}} = \sigma(X_{\text{in}} W_2 \odot X_{\text{in}} W_1) W_3
$$

where $X_{\text{in}} \in \R^{N\times T \times d_{\text{model}}}$, $W_{1,2} \in \R^{d_{\text{model}} \times d_{\text{hidden}}}$ and $W_{3} \in \R^{d_{\text{hidden}} \times d_{\text{model}}}$.

Now, lets work out the computational burden this function has for a forward pass!

- There are two operations that require $2NTd_{\text{model}}d_{\text{hidden}}$ FLOPs and memory $d_{\text{model}}d_{\text{hidden}}$
- An element-wise multiplication which is of the order $NTd_{\text{hidden}}$ and no memory overhead
- A final operation which requires a further $2NTd_{\text{model}}d_{\text{hidden}}$ and memory $d_{\text{model}}d_{\text{hidden}}$


So in total we have $6NTd_{\text{model}}d_{\text{hidden}}$ FLOPs.

By contrast, for the Attention part, we have the following formulation:

$$
A^h = \text{softmax} \left(\frac{X_{\text{in}}W^h_q \cdot {W_k^{h}}^T X_{\text{in}}^T}{\sqrt{d_h}}\right) \\
Y^h = A^h \cdot X_{\text{in}}W^h_v \\
X_{\text{out}} = Y^{h} W_{o}
$$

where $d_{\text{h}}$ is the attention head dimension, $W_{q,k,v}^h \in \R^{d_{\text{model}} \times d_{\text{h}}}$, $h$ is the attention head index which runs up to the total number of attention heads $H$ and $W_{o} \in \R^{H d_h \times d_{\text{model}}}$ .

- There are three multiplications in here, each requiring $2NTd_{\text{model}}d_{\text{h}}H$ and $Hd_{\text{model}}d_{\text{h}}$ memory.
- There is a dot product in the softmax operation which requires $2NT^2d_{h}H$ FLOPs (where we have a squared sequence length because we're creating the lookup table over all tokens in the sequence).
- Similarly, we have another dot product between the attention matrix and the value which involves $2NT^2d_{h}H$.
- Finally, we have the last matmul operation which multiplies and reduces all the attention head outputs into our final output, which requires $2NTd_{\text{model}}d_{\text{h}}H$ and $Hd_{\text{model}}d_{\text{h}}$ memory.

In total, that leaves us with $8NTd_{\text{model}}d_{\text{h}}H + 4NT^2d_{h}H$ FLOPs, where the first term is from the MLP block and the second is all down to attention.

Just to analyse this a bit more, lets investigate the relative difference between these two terms. Factoring common terms out we get:

$$
4NTd_{h}H(2d_{\text{model}} + T) \simeq 8NTd_{\text{model}}d_{\text{h}}H \qquad \text{when} \quad d_{\text{model}} \gg T/2
$$

where the MLPs dominate the FLOP count whenever $d_{\text{model}}$ is higher than the context size.

Right, so now lets just put this into context. Let's consider LLaMA 3-70B, which has $d_{model} = 8192$ and therefore we can get a pretty good approximation to the compute costs of this model whenever our sequence length is less than ~4k tokens!


### A quick note on training

The above was all done for inference, of course, for training things get slightly more complicated in terms of memory and FLOPs - especially when you start considering different checkpointing strategies for intermediate activations, different optimizer states, and other parallelisation techniques. However, we can make a simple adaptation to the FLOP calculations we did above but just considering the chain rule and backpropagation.

So, in training, gradients are essential for us to compute. Now imagine a set of feedforward layers stacked on top of each other. The goal of backpropagation is to compute a gradient with respect to the current layer weights (this is what's called a "leaf" in your computational graph), as well as the gradient with respect to the input of that layer.

Why do we need these? Well, the former we're going to use in our update equation to update the weights of that layer, and the latter we are going to pass down to the previous layer as our new vector in the vector-Jacobian-product chain.

Don't worry too much about the details of this here! All you need to really know is that there are 2 extra computations in the backwards pass, so you basically just want to add a factor of 3 to all the FLOP calculations we did above!

### Data Parallelism

In this setting, we shard activations along the batch dimension.

- There is no communication in the forward pass
- The gradients are calculated layer by layer and then asynchronously AllReduced while the rest of the backpropagation happens.
- These AllReduce operations are not in the critical path
- This is pretty forgiving since the communication costs can be overlapped with the computation - as long as the comms cost < compute cost.
- We can arbitrarily increase the batch size as long as we have more chips. In training, activations can dominate the memory footprint so this is very important.
- This does not reduce memory cost from the model or optimizer states.
- If the model & states don't fit on a single device, which is common, we cannot properly train with pure data parallelism.
- bf16 params and fp32 optimizer states (Adam has 2 moments), then we have 2 + 2 * 4 = 10 bytes per parameter.

When are we communication bottlenecked?
- We have two AllReduces per MLP block, each of size 2 * d_hidden * d_in.
- We have the per-chip FLOPs and the bandwidth. We can calculate the compute time and communication time.
- No need to do this for the forward pass, since there is no communication.
- An AllReduce time depends on the total bytes and the bandwidth available.
- The total time is the maximum of either the comms or compute time.
- To remain compute bound, we need the per-device batch size to exceed the operational intensity.
- Computation time scales with the per-device batch size, but the communication time is independent of this because we are transferring the model weights.

Context parallism:
- A batch is made up of sequences of tokens.
- We can do data parallelism across the batch and sequence dimension: which is called context parallelism,
- Attention is trickier because we do some cross-sequence computation.
- This can be handled by gathering KVs or Qs and overlapping compute and communication: ring attention.

### Fully-sharded Data Parallelism (FSDP/ZeRO-3)

The activations are still sharded along the batch dimension but also the rows of the parameters, gradients and optimizer states are sharded across the same axis.

- Model and optimizer states are sharded across data parallel shards
- FSDP drastically reduces per-device memory usage and saves backward pass FLOPs
- Couple of AllGathers in the forward pass to gather the sharded parameters before running the computation.
- These can be done while computing the previous layer activations.
- 2 ReduceScatters to reduce gradients across data shards scatter the gradients across devices in backward pass.
- 2 AllGathers in backward pass to gather the weight matrices to compute gradients across data sharded dimension.
- Also called ZeRO Sharding because we don't perform any unnecessary compute or store any unnecessary state.
- The 1, 2, 3 refers to the sharding of optimizer states, gradients and weights.
- All these have the same communication cost so we can always do ZeRO-3 sharding.
- Data parallelism does a lot of duplicated work i.e. AllReduces the full gradient, updates the optimizer state/parameters.
- Instead, we can do ReduceScatter on the gradients and update only a shard of the optimizer state/parameters. Then AllGather parameters in the forward pass.

When are we communication bottlenecked?
- The FLOPs and comms costs are exactly the same as the data parallelism case.
- However, the forward pass also has some communication in this case.
- But the communication bottleneck happens at the same point as in the data parallelism.
- So, as long as we satisfy the data parallelism bounds, we can upgrade to FSDP with no extra cost.
- Although we have added some communication cost to the forward pass, this overlaps with the compute.
- DeepSeek-v2 used a 40M batch size, which allows scaling to 47k chips or 5 TPUv5 pods before hitting a bandwidth limit.

Critical batch size:
- We become comms bottlenecked if our batch size decreases.
- Data Parallelism and FSDP allow us to scale to arbitraily many chips, as long as we increase our batch size.
- However, larger batch sizes make training see diminishing returns due to noise-free gradients.

### Model Parallelism

Here, activations and parameters are sharded across the model dimension. In between operations, there are AllGather and ReduceScatter operations.

- In FSDP we move weights across chips. We can also shard the feedforward dimension and move thr activations during the layer.
- This unlocks a smaller efficient batch size per pod.
- The forward pass steps include AllGathering the activations, performing the computation, then reduce scattering the output activations
- The backwards pass includes an AllGather on the output and saved input, then performing a ReduceScatter on the final gradient at the input.

### Mixing FSDP & Model Parallelism

- These two can be combined
- We can shard the weights along both axes and the batch along the first
- This reduces the size of the model AllGathers while also reducing the communication overhead of FSDP
- Combining the two can get to lower effective batch sizes
- FSDP moves the weights and model parallelism moves the activations
- As our batch size shrinks, the model parallelism becomes cheaper because our activations per-shard are smaller.
- Can get a rough factor of 2 smaller batch size than pure data parallel methods and still be compute bound

### Pipeline Parallelism

The idea of pipeline parallelism is to shard the model across its layer dimension.

- Split the layers across devices
- Pass activations between devices
- Do the same in reverse for the backwards pass
- This allows for training very large models, but leaves devices idle for long time periods
- Can mitigate this with microbatching
- Can also carefully overlap the forward matmuls and backward matmuls - typically by prioritising the backward pass derivative computations such to not block earlier devices and layers.
