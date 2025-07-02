MiniUMT: Lightweight Transformer for Micro-Expression Recognition
=================================================================

Overview
--------
This repository contains the code and utilities needed to train and evaluate MiniUMT, a compact transformer-based encoder optimized for detecting subtle human behaviors such as micro-expressions. Inspired by biological neuromodulation, MiniUMT integrates context-aware modulation into both its attention and feed-forward pathways to amplify fine-grained visual signals.

Repository Structure
--------------------
models/  
 Contains model definitions for MiniUMT and any supporting architectures.  

utils/  
 Utility functions for data loading, preprocessing, augmentation, logging, and metric computation.  

train.py  
 Entry point for training MiniUMT on your dataset.  

evaluate.py  
 Script to run inference and report performance metrics on a held-out test set.  

requirements.txt  
 List of Python dependencies needed to run training and evaluation scripts.  

Getting Started
---------------
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/MiniUMT.git
   cd MiniUMT
