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
    n: int  = 10 # Data length
    d: int  = 2 # Max polynomial degree
    r: int  = 5 # Privacy parameter
    p: int = 2 # Value for p-entropy
    epsilon: float = 1e-6 # smoothing "budget"
    a: float = 5 # confidence parameter of successful smoothing
    H_p_X: float  = None #p-entropy of the data
    max_H_p_X_R: float = None #Max p-entropy of any r-subset of the data
    m: int = None #Key length
    # Declared bound on |stored integer code|, i.e. X̂max = floor(A_X * 2^f_X) for quantised
    # data. Public, fixed before upload, and what the server's overflow check bounds queries
    # against (query.check_query). None means the data is native F_q data with no integer
    # meaning, so results are exact field values and overflow is not defined.
    max_input: int = None
