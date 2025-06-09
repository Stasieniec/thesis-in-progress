"""Model architectures for multimodal autism classification."""

try:
    from .fmri_transformer import SingleAtlasTransformer
    from .smri_transformer import SMRITransformer
    from .smri_transformer_working import WorkingNotebookSMRITransformer
    from .cross_attention import CrossAttentionTransformer as OriginalCrossAttentionTransformer
    from .improved_cross_attention import ImprovedCrossAttentionTransformer
    from .minimal_improved_cross_attention import MinimalImprovedCrossAttentionTransformer
    from .enhanced_smri import EnhancedSMRITransformer, SMRIEnsemble

    # Use minimal improved version as default
    CrossAttentionTransformer = MinimalImprovedCrossAttentionTransformer

    __all__ = [
        "SingleAtlasTransformer",
        "SMRITransformer", 
        "WorkingNotebookSMRITransformer",
        "EnhancedSMRITransformer",
        "SMRIEnsemble",
        "CrossAttentionTransformer",
        "OriginalCrossAttentionTransformer",
        "ImprovedCrossAttentionTransformer",
        "MinimalImprovedCrossAttentionTransformer"
    ]
except ImportError:
    # Graceful fallback if modules don't exist
    try:
        from .cross_attention import CrossAttentionTransformer
        __all__ = ["CrossAttentionTransformer"]
    except ImportError:
        __all__ = [] 