# 🚀 Advanced Cross-Attention Experiments Guide

This guide explains how to use the advanced cross-attention experiments script to systematically test different strategies for beating your fMRI baseline of 65%.

## 🎯 Goal

Your current results:
- **fMRI baseline**: 65% accuracy (target to beat)
- **sMRI baseline**: 58% accuracy 
- **Current cross-attention**: 63.6% accuracy (needs improvement)

**Objective**: Find a cross-attention strategy that consistently beats 65% fMRI accuracy.

## 🧠 Available Strategies

### 1. **Bidirectional Cross-Attention** (`bidirectional`)
- **Strategy**: Bidirectional attention flow (fMRI ↔ sMRI)
- **Key Features**:
  - Multiple layers of bidirectional cross-attention
  - Enhanced normalization and residual connections
  - Multi-layer attention processing
- **Best for**: Learning complex interaction patterns between modalities

### 2. **Hierarchical Cross-Attention** (`hierarchical`)
- **Strategy**: Multi-scale processing with hierarchical fusion
- **Key Features**:
  - Processes features at 3 different scales simultaneously
  - Scale-specific cross-attention mechanisms
  - Hierarchical feature fusion
- **Best for**: Capturing features at different granularities

### 3. **Contrastive Cross-Attention** (`contrastive`)
- **Strategy**: Contrastive learning for better modal alignment
- **Key Features**:
  - Contrastive projections for modal alignment
  - Learned attention temperature
  - Residual connection from strong modality (fMRI)
- **Best for**: Learning better cross-modal representations

### 4. **Adaptive Cross-Attention** (`adaptive`)
- **Strategy**: Dynamic gating and adaptive temperature
- **Key Features**:
  - Adaptive temperature learning based on input
  - Dynamic modality importance gating
  - Performance-aware weighting (65% vs 58%)
- **Best for**: Automatically adapting to input characteristics

### 5. **Ensemble Cross-Attention** (`ensemble`)
- **Strategy**: Ensemble of multiple attention mechanisms
- **Key Features**:
  - Multiple parallel attention mechanisms
  - Learnable ensemble weights
  - Meta-classifier for combination
- **Best for**: Robust performance through diversity

## 🚀 Usage

### Quick Test (Fast Development)
```python
# Test the most promising strategy quickly (2 folds, 10 epochs)
!python scripts/advanced_cross_attention_experiments.py quick_test

# Test a specific strategy quickly
!python scripts/advanced_cross_attention_experiments.py quick_test --strategy=adaptive
```

### Test Individual Strategy
```python
# Test bidirectional cross-attention (full evaluation)
!python scripts/advanced_cross_attention_experiments.py test_strategy --strategy=bidirectional

# Test adaptive strategy with custom parameters
!python scripts/advanced_cross_attention_experiments.py test_strategy \
    --strategy=adaptive \
    --num_epochs=150 \
    --learning_rate=2e-5 \
    --d_model=256

# Test hierarchical strategy
!python scripts/advanced_cross_attention_experiments.py test_strategy --strategy=hierarchical
```

### Run All Strategies (Comprehensive Evaluation)
```python
# Run all strategies with default parameters
!python scripts/advanced_cross_attention_experiments.py run_all

# Run all strategies with custom parameters
!python scripts/advanced_cross_attention_experiments.py run_all \
    --num_epochs=200 \
    --batch_size=32 \
    --learning_rate=3e-5 \
    --d_model=256 \
    --num_folds=5
```

## 📊 Understanding Results

The script will output a comprehensive comparison showing which strategies beat your baselines.

Good luck beating that fMRI baseline! 🎯🚀 