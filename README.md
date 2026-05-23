# AI-Secure-WBAN-for-ETSI-Standard 🏥🛡️

This repository contains the implementation of my Master's Thesis: **"Physical and Link Layer Security of Wireless Body Area Networks with Deep Learning Model-Based Applications"** conducted at **Politecnico di Milano**.

---

## 📄 Unified Project Documentation & Execution Script

The documentation below provides a high-fidelity technical breakdown of the **ETSI SmartBAN** simulation framework and the AI-driven security analysis, followed by the complete execution script.

```text
================================================================================
1. RESEARCH CONTEXT & ACADEMIC REFERENCE
================================================================================
Institution:   Politecnico di Milano (1863)
Author:        Berk Aybey
Advisor:       Prof. Maurizio Magarini
Focus:         Security at PHY and Link layers for ETSI SmartBAN systems.

================================================================================
2. PROJECT ARCHITECTURE (AI-Secure-WBAN-for-ETSI-Standard)
================================================================================
* /Simulator: 
    - Implements ETSI SmartBAN PHY/MAC layers (TS 103 327 / TR 103 711).
    - Features GFSK modulation, Ricean fading models, and BCH/CRC error control.
    - Includes modules for attack injection (Jamming, Replay, Packet Injection).
* /Deep-Learning:
    - AI Pipeline: Processes 1.5M+ binary samples with 96 features.
    - Architectures: LSTM (Best F1-Score), GRU, RNN, 1D-CNN, MLP.
    - /HyperparameterTuning: DEAP-based Genetic Algorithm (GA) optimization.

================================================================================
3. AI PIPELINE & PERFORMANCE METRICS
================================================================================
* Results: Achieved 81% F1-score with optimized LSTM architecture.
* Latency: Real-time monitoring capability with processing times < 250ms.
* Methodology: Evolutionary hyperparameter search ensures robust security 
               detection under varying channel conditions.

================================================================================
4. INSTALLATION & EXECUTION BASH SCRIPT
================================================================================
# 1. Clone the repository
git clone [https://github.com/BMrMoon/AI-Secure-WBAN-for-ETSI-Standard.git](https://github.com/BMrMoon/AI-Secure-WBAN-for-ETSI-Standard.git)
cd AI-Secure-WBAN-for-ETSI-Standard

# 2. Setup Conda Environment
conda create --name wban_env python=3.9.19 -y
conda activate wban_env

# 3. Install dependencies
conda install --file requirements.txt

# 4. Launch Simulator (GUI)
python Simulator/gui/main_gui.py

# 5. Execute Genetic Optimizer
# Now located at: Deep-Learning/HyperparameterTuning/run.py
python Deep-Learning/HyperparameterTuning/run.py
================================================================================
