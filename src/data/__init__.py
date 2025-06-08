"""Data processing modules for fMRI and sMRI data."""

try:
    from .fmri_processor import FMRIDataProcessor
    from .smri_processor import SMRIDataProcessor, SMRIDataset
    from .multimodal_dataset import MultiModalDataset, MultiModalPreprocessor, match_multimodal_subjects
    from .base_dataset import ABIDEDataset
    # Note: site_aware_data_loader import removed to prevent import errors
    
    __all__ = [
        "FMRIDataProcessor",
        "SMRIDataProcessor",
        "SMRIDataset", 
        "MultiModalDataset",
        "MultiModalPreprocessor",
        "match_multimodal_subjects",
        "ABIDEDataset"
    ]
except ImportError:
    # Graceful fallback if modules don't exist
    __all__ = [] 