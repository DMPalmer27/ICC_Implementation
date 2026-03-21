from dataclasses import dataclass

@dataclass
class SystemContext:
    q: int  # Field size
    n: int  # Data length
    m: int  # Key length
    d: int  # Max polynomial degree
    r: int  # Privacy parameter