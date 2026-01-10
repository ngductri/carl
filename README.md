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

---

## 🛠️ Setup Instructions

Follow the steps below to install Miniconda, configure the environment, and set up CARLA.

### **1. Install Miniconda**
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
