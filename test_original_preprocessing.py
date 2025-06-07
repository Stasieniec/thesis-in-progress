#!/usr/bin/env python3
"""
Test cross-attention with ORIGINAL sMRI preprocessing.
This should help diagnose if the issue is preprocessing mismatch.
"""

# Modify train_cross_attention.py to use:
# - StandardScaler instead of RobustScaler  
# - Basic feature selection instead of F-score + MI
# - Original cross-attention architecture

print("🧪 Testing original preprocessing hypothesis...")
print("Expected: Recovery to ~63.6% if preprocessing was the issue")
