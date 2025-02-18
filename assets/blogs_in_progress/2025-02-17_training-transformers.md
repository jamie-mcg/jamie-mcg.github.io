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

## An MLP Perspective

To begin, we will consider parallelisation of an MLP 

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
- $$

### Pipeline Parallelism

The idea of pipeline parallelism is to shard the model across its layer dimension.
