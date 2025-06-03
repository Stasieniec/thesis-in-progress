"""sMRI data processing module."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif

from data.base_dataset import ABIDEDataset


class SMRIDataProcessor:
    """Process pre-processed sMRI data from FreeSurfer features."""
    
    def __init__(self, data_path: Path, feature_selection_k: Optional[int] = None, scaler_type: str = 'robust'):
        """
        Initialize sMRI data processor.
        
        Args:
            data_path: Path to processed sMRI data directory
            feature_selection_k: Number of features to select (None for all)
            scaler_type: Type of scaler ('robust' or 'standard')
        """
        self.data_path = data_path
        self.feature_selection_k = feature_selection_k
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.feature_names = None

    def load_smri_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Load pre-processed sMRI data.
        
        Returns:
            features: Feature array
            labels: Label array  
            subject_ids: Subject ID array
            feature_names: List of feature names
        """
        features = np.load(self.data_path / 'features.npy')
        labels = np.load(self.data_path / 'labels.npy')
        subject_ids = np.load(self.data_path / 'subject_ids.npy')

        with open(self.data_path / 'feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]

        return features, labels, subject_ids, feature_names

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit preprocessors on training data.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            Transformed feature array
        """
        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Initialize scaler
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)

        # Feature selection if specified
        if self.feature_selection_k and self.feature_selection_k < X.shape[1]:
            self.feature_selector = SelectKBest(score_func=f_classif, k=self.feature_selection_k)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            self.selected_features = self.feature_selector.get_support()
            print(f"Selected {self.feature_selection_k} best features out of {X.shape[1]}")
            return X_selected
        else:
            return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted preprocessors.
        
        Args:
            X: Feature array
            
        Returns:
            Transformed feature array
        """
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        if self.feature_selector is not None:
            return self.feature_selector.transform(X_scaled)
        return X_scaled

    def analyze_features(self, X: np.ndarray, y: np.ndarray, top_k: int = 200) -> pd.DataFrame:
        """
        Analyze feature importance.
        
        Args:
            X: Feature array
            y: Label array
            top_k: Number of top features to analyze
            
        Returns:
            DataFrame with feature importance statistics
        """
        print(f"Analyzing feature importance...")

        # Calculate F-scores
        f_scores, f_pvals = f_classif(X, y)

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature_name': self.feature_names,
            'f_score': f_scores,
            'f_pval': f_pvals,
        })

        # Sort by F-score
        feature_importance = feature_importance.sort_values('f_score', ascending=False)

        print(f"Top 10 most important features:")
        print(feature_importance.head(10)[['feature_name', 'f_score', 'f_pval']].to_string(index=False))

        return feature_importance

    def process_all_subjects(self, phenotypic_file: Path, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Process all sMRI subjects with label correction.
        
        Args:
            phenotypic_file: Path to phenotypic data file
            verbose: Whether to print statistics
            
        Returns:
            features: Processed feature array
            labels: Corrected label array (0=Control, 1=ASD)
            subject_ids: List of subject IDs
        """
        # Load sMRI data
        features, _, subject_ids, feature_names = self.load_smri_data()
        self.feature_names = feature_names
        
        # Load phenotypic data for label correction
        pheno = pd.read_csv(phenotypic_file)
        pheno['SUB_ID'] = pheno['SUB_ID'].astype(str)
        
        # Create mapping from phenotypic data
        pheno_dict = {}
        for _, row in pheno.iterrows():
            if row['DX_GROUP'] in [1, 2]:
                sub_id = str(row['SUB_ID'])
                # Use same convention as fMRI: 0=Control, 1=ASD
                pheno_dict[sub_id] = 0 if row['DX_GROUP'] == 2 else 1

        # Correct labels and filter subjects
        corrected_features = []
        corrected_labels = []
        corrected_subject_ids = []
        
        for i, sub_id in enumerate(subject_ids):
            # Handle both string and integer subject IDs
            if isinstance(sub_id, (int, np.integer)):
                sub_id_str = str(sub_id)
            else:
                try:
                    sub_id_str = str(int(sub_id))
                except:
                    sub_id_str = str(sub_id)
            
            if sub_id_str in pheno_dict:
                corrected_features.append(features[i])
                corrected_labels.append(pheno_dict[sub_id_str])
                corrected_subject_ids.append(sub_id_str)
        
        corrected_features = np.array(corrected_features)
        corrected_labels = np.array(corrected_labels)
        
        if verbose:
            print(f"✅ Processed {len(corrected_labels)} sMRI subjects")
            print(f"🧠 ASD: {np.sum(corrected_labels)}, Control: {len(corrected_labels) - np.sum(corrected_labels)}")
        
        return corrected_features, corrected_labels, corrected_subject_ids

    def load_all_subjects(self, phenotypic_file: Path) -> Dict:
        """
        Load all sMRI subjects and return as dictionary (for cross-modal matching).
        
        Args:
            phenotypic_file: Path to phenotypic data file
            
        Returns:
            Dictionary mapping subject IDs to features and labels
        """
        features, labels, subject_ids = self.process_all_subjects(phenotypic_file, verbose=False)
        
        smri_data = {}
        for i, sub_id in enumerate(subject_ids):
            smri_data[sub_id] = {
                'features': features[i],
                'label': labels[i]
            }
        
        return smri_data


class SMRIDataset(ABIDEDataset):
    """sMRI-specific dataset class."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[List[str]] = None,
        augment: bool = False,
        noise_factor: float = 0.005
    ):
        """
        Initialize sMRI dataset.
        
        Args:
            features: sMRI feature array
            labels: Label array
            subject_ids: Optional list of subject IDs
            augment: Whether to apply data augmentation
            noise_factor: Noise factor for augmentation (smaller for sMRI)
        """
        super().__init__(
            X=features,
            y=labels,
            subject_ids=subject_ids,
            augment=augment,
            noise_std=noise_factor,
            augment_prob=0.5  # Higher probability for sMRI
        ) 