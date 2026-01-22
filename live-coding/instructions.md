# Live Coding + Debugging Interview – Instructions

**Duration:** 60–90 minutes  
**Target Role:** Machine Learning / Applied AI Engineer  
**Focus:** Practical ML coding, debugging, and inference pipeline skills

---

## Overview of Tasks

The session consists of four tasks:

1. **Time-Series Feature Engineering (Task 1)**  
   Implement a function that builds sliding windows and labels from a timestamped series.

2. **Training Loop with Early Stopping (Task 2)**  
   Implement a simple training loop that monitors validation loss and applies early stopping.

3. **Debug a PyTorch Training Snippet (Task 3)**  
   Identify and fix bugs in a deliberately broken PyTorch training loop.

4. **Inference Batching with CPU/GPU Switching (Task 4)**  
   Implement a batched inference utility that transparently runs on CPU or GPU.

---

## Candidate Instructions (To Be Read at Start)

> Today we will run a live coding and debugging session focused on applied machine learning engineering.  
> We will go through four tasks that simulate typical production problems:
> 
> 1. Time-series feature engineering  
> 2. Training loop with early stopping  
> 3. Debugging a PyTorch training script  
> 4. Implementing batched inference with CPU/GPU switching
> 
> Please **think aloud** as much as you can. We are interested in your reasoning, not just the final answer.  
> You may assume you have access to standard Python libraries such as:
> 
> - `numpy`, `pandas`  
> - `scikit-learn`  
> - `torch` (PyTorch)
> 
> You may use any coding style you are comfortable with, but aim for clarity and correctness.  
> If something is unclear in a task, you can ask for clarification.

---

## Task Files

- `task1_time_series.py` – Time-series feature engineering.
- `task1_sample_data.csv` – Sample input data for Task 1 (optional to use live).
- `task2_early_stopping.py` – Training loop with early stopping.
- `task3_debug_pytorch.py` – Debugging PyTorch training loop.
- `task4_inference_batching.py` – Batched inference with CPU/GPU switching.

---

