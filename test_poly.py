import numpy as np

# A payload: 1 0 1 1 0
bits = np.array([1, 0, 1, 1, 0])
# CRC-3 polynomial: x^3 + x + 1 -> 1 0 1 1
poly = np.array([1, 0, 1, 1])

# Pad bits with 3 zeros
padded = np.concatenate((bits, [0, 0, 0]))

# Polydiv in GF(2)
q, r = np.polydiv(padded, poly)
r = np.abs(np.round(r)).astype(int) % 2
print(r)
