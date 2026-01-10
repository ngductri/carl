# 🚗 CaRL – Customized Autonomous RL Pipeline
*Forked & extended from the official CaRL repository*

---

## 📌 Introduction  
This repository is a **customized version of the CaRL (CARLA Reinforcement Learning) project**, developed by **our research team**.  

The original CaRL project can be found here:

[![GitHub – Original CaRL](https://img.shields.io/badge/GitHub-Original%20CaRL-black?logo=github)](https://github.com/autonomousvision/CaRL)

Our fork is designed to **extend, improve, and experiment** beyond the original implementation. Key goals of our modifications include:

- Exploring how different **visual encoders** affect agent performance  
- Replacing the original **PPO RL algorithm** with **SAC (Soft Actor-Critic)** to study the impact of off-policy methods  
- Modifying and tuning various **hyperparameters** to optimize training efficiency and stability  
- Enhancing **logging, debugging, and evaluation tools** for more granular analysis  
- Maintaining the core CaRL pipeline while increasing **modularity** and **development flexibility**

> ⚠️ **Note:** This is *not* the official CaRL repository.  
> It is a **research-oriented fork** tailored to our experiments in autonomous driving.


🔍 Key Custom Modifications

Compared to the original CaRL repository, our fork introduces several research-focused modifications:

Encoder Upgrades

Replaced the original BEV (bird’s-eye-view) encoder with state-of-the-art vision backbones such as ResNet50 and MobileNetV3.

Purpose: Evaluate how encoder capacity affects policy learning and driving performance of the agent.

RL Algorithm Replacement

Original CaRL uses PPO. We implemented Soft Actor-Critic (SAC) as an alternative.

SAC is an off-policy, entropy-regularized algorithm, which can improve sample efficiency and robustness in continuous control tasks.

This allows systematic comparison between on-policy (PPO) and off-policy (SAC) methods for autonomous driving.

Hyperparameter Modifications

Adjusted learning rates, batch sizes, discount factors, and exploration noise to better suit the modified encoders and SAC algorithm.

This ensures more stable training and allows deeper analysis of encoder-policy interactions.

Logging & Evaluation Enhancements

Improved logging to track per-step rewards, collisions, success rates, and other driving metrics.

Added richer evaluation tools for visualizing agent trajectories, comparing encoder performance, and analyzing failure cases.

These modifications allow researchers to systematically analyze the effects of visual representations, RL algorithms, and hyperparameters on autonomous driving performance in CARLA.


