"""Model architectures for multimodal autism classification."""

from models.fmri_transformer import SingleAtlasTransformer
from models.smri_transformer import SMRITransformer
from models.smri_transformer_working import WorkingNotebookSMRITransformer
from models.cross_attention import CrossAttentionTransformer as OriginalCrossAttentionTransformer
from models.improved_cross_attention import ImprovedCrossAttentionTransformer
from models.minimal_improved_cross_attention import MinimalImprovedCrossAttentionTransformer

# Use minimal improved version as default
CrossAttentionTransformer = MinimalImprovedCrossAttentionTransformer

__all__ = [
    "SingleAtlasTransformer",
    "SMRITransformer", 
    "WorkingNotebookSMRITransformer",
    "CrossAttentionTransformer",
    "OriginalCrossAttentionTransformer",
    "ImprovedCrossAttentionTransformer",
    "MinimalImprovedCrossAttentionTransformer"
] 