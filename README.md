# ABIDE Multimodal Transformer

A comprehensive framework for multimodal autism classification using fMRI and sMRI data from the ABIDE dataset. This repository implements state-of-the-art transformer architectures including single-modality transformers and cross-attention mechanisms for multimodal learning.

## 🧠 Overview

This repository contains implementations of:

- **Single Atlas Transformer (SAT)**: Enhanced transformer for fMRI functional connectivity data
- **sMRI Transformer**: Transformer architecture for structural MRI features  
- **Cross-Attention Transformer**: Multimodal transformer with cross-modal attention between fMRI and sMRI

All models are designed for binary classification of autism spectrum disorder (ASD) vs. typical controls using the ABIDE dataset.

## 🏗️ Architecture

```
thesis-in-progress/
├── src/                     # Source code
│   ├── config/             # Configuration classes
│   ├── data/               # Data processing modules
│   ├── models/             # Model architectures
│   ├── training/           # Training framework
│   ├── evaluation/         # Evaluation metrics
│   └── utils/              # Utility functions
├── scripts/                # Fire-based CLI scripts
│   ├── train_fmri.py       # fMRI-only experiments
│   ├── train_smri.py       # sMRI-only experiments
│   └── train_cross_attention.py  # Multimodal experiments
├── context_files/          # Additional context files
├── data/                   # Data directory
└── requirements.txt        # Dependencies
```

## 🚀 Features

- **Modular Design**: Clean separation of data processing, models, training, and evaluation
- **Configuration Management**: Dataclass-based configurations for different experiment types
- **Fire CLI**: Easy command-line interface for running experiments
- **Cross-Validation**: Robust k-fold stratified cross-validation
- **Mixed Precision Training**: Efficient GPU utilization with automatic mixed precision
- **Comprehensive Evaluation**: Multiple metrics, visualizations, and statistical analysis
- **Reproducibility**: Seed management and deterministic training

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- See `requirements.txt` for complete dependencies

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd thesis-in-progress
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python scripts/train_fmri.py get_config_template
```

## 📊 Data Requirements

The framework expects data in the following structure:

### fMRI Data
- ROI time series files in `/content/drive/MyDrive/b_data/ABIDE_pcp/cpac/filt_noglobal/rois_cc200/`
- Phenotypic file: `/content/drive/MyDrive/b_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv`

### sMRI Data  
- Processed FreeSurfer features in `/content/drive/MyDrive/processed_smri_data/`
- Required files: `features.npy`, `labels.npy`, `subject_ids.npy`, `feature_names.txt`

## 🎯 Usage

### fMRI-only Experiments

```bash
# Basic run with default parameters
python scripts/train_fmri.py run

# Custom configuration
python scripts/train_fmri.py run --num_folds=10 --batch_size=128 --learning_rate=5e-5

# Quick test (for debugging)
python scripts/train_fmri.py quick_test
```

### sMRI-only Experiments

```bash
# Basic run
python scripts/train_smri.py run

# With feature selection
python scripts/train_smri.py run --feature_selection_k=500 --batch_size=32

# Feature analysis only
python scripts/train_smri.py analyze_features_only --top_k=50
```

### Cross-Attention Multimodal Experiments

```bash
# Basic multimodal run
python scripts/train_cross_attention.py run

# Custom architecture
python scripts/train_cross_attention.py run --d_model=512 --num_cross_layers=4

# Analyze data overlap
python scripts/train_cross_attention.py analyze_data_overlap
```

## ⚙️ Configuration

Each experiment type has its own configuration class with sensible defaults:

### fMRI Configuration
- Embedding dimension: 256
- Transformer layers: 4  
- Attention heads: 8
- Batch size: 256
- Learning rate: 1e-4

### sMRI Configuration  
- Embedding dimension: 64
- Transformer layers: 2
- Feature selection: 300 top features
- Batch size: 16
- Learning rate: 1e-3

### Cross-Attention Configuration
- Embedding dimension: 256
- Cross-attention layers: 2
- Batch size: 32
- Learning rate: 5e-5

## 📈 Model Architectures

### Single Atlas Transformer (SAT)
- Input: fMRI connectivity features (19,900 dimensions for CC200)
- Architecture: Enhanced transformer with pre-normalization
- Output: Binary classification (ASD vs. Control)

### sMRI Transformer
- Input: Selected structural features (default: 300)
- Architecture: Lightweight transformer with CLS token
- Regularization: High dropout, batch normalization

### Cross-Attention Transformer
- Inputs: Both fMRI and sMRI features
- Architecture: Modality-specific encoders + cross-attention layers
- Fusion: Late fusion of CLS tokens

## 🔬 Evaluation

The framework provides comprehensive evaluation including:

- **Metrics**: Accuracy, balanced accuracy, AUC, precision, recall, F1
- **Cross-Validation**: Stratified k-fold with proper preprocessing
- **Visualizations**: Learning curves, confusion matrices, ROC curves
- **Statistical Analysis**: Mean ± std, confidence intervals

## 📁 Output Structure

Results are saved in timestamped directories:

```
output_directory/
├── fold1_results.json      # Individual fold results
├── fold2_results.json
├── ...
├── experiment_results.json # Complete results
├── experiment_cv_results.png  # Visualizations
└── best_model_fold*.pt     # Saved models
```

## 🔧 Development

### Adding New Models

1. Create model in `src/models/`
2. Add import to `src/models/__init__.py`
3. Create configuration in `src/config/config.py`
4. Add script in `scripts/`

### Adding New Features

The modular design makes it easy to extend:

- **Data processors**: Add to `src/data/`
- **Training strategies**: Modify `src/training/trainer.py`
- **Evaluation metrics**: Extend `src/evaluation/metrics.py`

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{abide_multimodal_transformer,
  title={ABIDE Multimodal Transformer Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Contact

For questions or issues, please open a GitHub issue or contact [your-email].

---

**Note**: This framework is designed for research purposes. Ensure you have appropriate permissions to use the ABIDE dataset and follow all relevant data usage agreements.