"""
File: config.py
Author: Daniel Palmer (d.m.palmer@wustl.edu)
Description: This file defines the SystemContext object which holds parameters and useful information
for the entire scheme instance
"""

from dataclasses import dataclass

@dataclass
class SystemContext:
    q: int = 31 # Field size
    n: int  = 30 # Data length
    d: int  = 2 # Max polynomial degree
    r: int  = 1 # Privacy parameter
    p: int = 2 # Value for p-entropy
    epsilon: float = 1e-6 # smoothing "budget"
    a: float = 20 # confidence parameter of successful smoothing
    H_p_X: float  = None #p-entropy of the data
    max_H_p_X_R: float = None #Max p-entropy of any r-subset of the data
    m: int = None #Key length
