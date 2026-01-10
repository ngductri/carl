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
This repository is a **research-oriented fork** of the official CaRL project.  
We extend and modify the original pipeline to **experiment with different encoders, RL algorithms, and hyperparameters**, while keeping the core modular structure.

**Our main goals:**
- Evaluate how **SOTA visual encoders** affect agent performance  
- Replace PPO with **Soft Actor-Critic (SAC)** for improved off-policy learning  
- Enhance **logging, evaluation, and debugging tools**  
- Tune hyperparameters for **stable and efficient training**

> ⚠️ *Not the official CaRL repository – intended for research and experimentation.*

---

## 🔍 Key Custom Modifications

| Component | Original CaRL | Our Fork |
|-----------|---------------|----------|
| Encoder | Basic BEV encoder | ResNet50, MobileNetV3 (SOTA) |
| RL Algorithm | PPO (on-policy) | SAC (off-policy, entropy-regularized) |
| Hyperparameters | Default | Tuned learning rates, batch sizes, exploration noise, discount factor |
| Logging & Evaluation | Basic | Detailed per-step metrics, trajectory visualization, encoder comparison |

