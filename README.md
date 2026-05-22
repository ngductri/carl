# 🚗 CaRL – Customized Autonomous RL Pipeline
*Forked & extended from the official CaRL repository*

[![GitHub – Original CaRL](https://img.shields.io/badge/GitHub-Original%20CaRL-black?logo=github)](https://github.com/autonomousvision/CaRL)

---

## 📌 Table of Contents
- [Introduction](#introduction)  
- [Key Custom Modifications](#key-custom-modifications)  
- [Research Policy Optimization (RPO)](#research-policy-optimization-rpo)  
- [Training & Evaluation](#training--evaluation)  
- [Notes & Recommendations](#notes--recommendations)  
- [References](#references)

---

## 📌 Introduction

This repository is a **research-focused fork** of the official CaRL (CARLA Reinforcement Learning) project.  
Our goal is to explore how different components of an autonomous driving RL pipeline—**perception, policy, and hyperparameters**—interact and affect performance in realistic driving scenarios.

Key objectives:

1. Evaluate the impact of **SOTA visual encoders** (ResNet50, MobileNetV3) on agent performance  
2. Replace PPO with **Soft Actor-Critic (SAC)** for off-policy training stability  
3. Tune hyperparameters and apply RPO for better convergence and sample efficiency  
4. Improve logging and evaluation for systematic analysis  

> ⚠️ This fork is **not the official CaRL repository**. It is intended for **research and experimentation**.

---

## 🔍 Key Custom Modifications

| Component | Original CaRL | Our Fork | Purpose / Benefits |
|-----------|---------------|----------|------------------|
| **Encoder** | Basic BEV encoder | ResNet50, MobileNetV3 | Measure perception impact on policy; evaluate trade-offs between complexity and performance |
| **RL Algorithm** | PPO (on-policy) | SAC (off-policy) | More stable off-policy learning; better sample efficiency |
| **Hyperparameters** | Default | Tuned learning rate, batch size, discount factor, exploration noise | Stabilize training across encoders and algorithms |
| **Logging & Evaluation** | Basic logging | Detailed per-step metrics, trajectory visualization, encoder comparison | Facilitate research and debugging |

**Workflow Overview:**

1. **Observation:** Multi-modal input (BEV semantic map + vehicle state)  
2. **Policy:** Encoded observation → RL policy (SAC/PPO) → continuous actions  
3. **Environment:** Actions applied in CARLA → next observation, reward, done flag  
4. **Training Loop:** SAC updates off-policy; metrics logged  
5. **Evaluation:** Metrics: success rate, collisions, lane infractions, route completion; optional trajectory visualization
