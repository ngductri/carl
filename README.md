# 🚗 CaRL – Customized Autonomous RL Pipeline
*Forked & extended from the official CaRL repository*

[![GitHub – Original CaRL](https://img.shields.io/badge/GitHub-Original%20CaRL-black?logo=github)](https://github.com/autonomousvision/CaRL)


---

## 📌 Table of Contents
- [Introduction](#introduction)  
- [Key Custom Modifications](#key-custom-modifications)  
- [Training & Evaluation](#training--evaluation)  
- [Notes & Recommendations](#notes--recommendations)  
- [References](#references)

---

## 📌 Introduction

This repository is a **research-focused fork** of the official CaRL (CARLA Reinforcement Learning) project.  
Our goal is to explore how different components of an autonomous driving RL pipeline—**perception, policy, and hyperparameters**—interact and affect performance in realistic driving scenarios.

While the original CaRL pipeline provides a robust framework for autonomous driving RL, we have introduced several key improvements and extensions aimed at:

1. **Understanding the impact of visual perception on policy learning:**  
   - By replacing the original BEV (bird’s-eye-view) encoder with **state-of-the-art convolutional backbones** like **ResNet50** and **MobileNetV3**, we can systematically evaluate how encoder capacity influences the agent's ability to perceive and reason about its environment.

2. **Exploring alternative reinforcement learning algorithms:**  
   - The original CaRL uses **PPO**, an on-policy algorithm.  
   - We implement **Soft Actor-Critic (SAC)**, an **off-policy, entropy-regularized RL algorithm**, which provides better sample efficiency and more stable exploration in continuous control tasks.  
   - This allows us to compare the effects of **on-policy vs off-policy training** on driving performance.

3. **Fine-tuning hyperparameters for research:**  
   - Adjustments include learning rate, batch size, discount factor, and exploration noise.  
   - These changes are designed to stabilize training across different encoders and RL algorithms, allowing fair comparisons and deeper analysis.

4. **Improved logging and evaluation:**  
   - Detailed per-step metrics such as reward, collisions, lane departures, and route completion.  
   - Visualization tools for trajectory analysis, encoder comparison, and failure case study.  

> ⚠️ This fork is **not the official CaRL repository**.  
> It is intended for **research, experimentation, and systematic analysis** of autonomous driving RL.

---

## 🔍 Key Custom Modifications

| Component | Original CaRL | Our Fork | Purpose / Benefits |
|-----------|---------------|----------|------------------|
| **Encoder** | Basic BEV encoder | ResNet50, MobileNetV3 (SOTA) | Evaluate perception capacity on policy learning; analyze trade-off between encoder complexity and agent performance |
| **RL Algorithm** | PPO (on-policy) | SAC (off-policy, entropy-regularized) | Off-policy training improves sample efficiency; allows more stable exploration in continuous control |
| **Hyperparameters** | Default | Tuned learning rates, batch sizes, exploration noise, discount factor | Optimized for different encoders and RL algorithms; improves stability and convergence speed |
| **Logging & Evaluation** | Basic logging | Detailed per-step metrics, trajectory visualization, encoder comparison | Facilitates research analysis; easier debugging and performance comparison across experiments |

### 🔹 Workflow Overview

1. **Observation**  
   - Agent receives a **multi-modal observation**:
     - BEV semantic map (processed by chosen encoder)  
     - Low-dimensional vehicle state measurements (speed, steering, etc.)  

2. **Policy**  
   - Encoded observation is passed to the RL policy (PPO or SAC).  
   - SAC policy outputs continuous actions: throttle, brake, steering.  

3. **Environment Interaction**  
   - Actions applied in CARLA simulator.  
   - Environment returns next observation, reward, and done flag.  

4. **Training Loop**  
   - SAC updates policy off-policy using replay buffer.  
   - Metrics are logged for each timestep to analyze performance trends.

5. **Evaluation**  
   - Evaluate trained policies on predefined towns/routes.  
   - Metrics include **success rate, collisions, lane infractions, route completion**, plus trajectory visualizations.

This structured pipeline allows us to **isolate the effect of encoders, RL algorithms, and hyperparameters** in controlled experiments.
