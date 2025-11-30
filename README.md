# 🚗 CaRL – Customized Autonomous RL Pipeline
*Forked & extended from the official CaRL repository*

---

## 📌 Introduction  
This project is developed by **our team** as part of our work on the **CaRL (CARLA Reinforcement Learning)** system.  
The current repository is a **fork of the official CaRL project**:

[![GitHub – Original CaRL](https://img.shields.io/badge/GitHub-Original%20CaRL-black?logo=github)](https://github.com/autonomousvision/CaRL)

We use this fork to:

- Extend and modify the original implementation for research purposes  
- Experiment with new and customized Reinforcement Learning algorithms  
- Improve logging, debugging workflows, and evaluation tools  
- Keep the core structure while increasing modularity and development flexibility  

> ⚠️ **Note:** This is *not* the official CaRL repository.  
> It is a customized internal version tailored to our experiments.

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
```

2. Clone & Enter the CaRL Project
bash
Copy code
# sudo apt install git   # optional if git is not installed
cd CaRL/CARLA
3. Create Conda Environment
bash
Copy code
conda env create -f environment.yml
conda activate carl
4. Setup CARLA
bash
Copy code
bash setup_carla.sh
Your environment is now ready for training and evaluation.
