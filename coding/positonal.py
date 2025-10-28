# %%
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_positional_encoding(n, d, base=10000.0):
    pos = np.arange(n)[:, None]                 # t = 0..n-1
    i = np.arange(d // 2)[None, :]              # i = 0..d/2-1
    angles = pos * (1.0 / (base ** (2 * i / d)))
    pe = np.zeros((n, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles)                # Φ_{t,2i}
    pe[:, 1::2] = np.cos(angles)                # Φ_{t,2i+1}
    return pe

n, d, base = 128, 64, 10000.0
pe = sinusoidal_positional_encoding(n, d, base)
pe = pe.transpose(1, 0)

plt.figure(figsize=(8, 4))
im = plt.imshow(pe, aspect='auto', origin='lower', cmap='twilight')  # 不指定颜色映射

# y轴坐标值要reverse，从上到下为从小到大

plt.xlabel('position t')
plt.ylabel('dimension (2i / 2i+1)')
plt.title(f'Sinusoidal Positional Encoding Heatmap (n={n}, d={d}, base={int(base)})')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

# %%
