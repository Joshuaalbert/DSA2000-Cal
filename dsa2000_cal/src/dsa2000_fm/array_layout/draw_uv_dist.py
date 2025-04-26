import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0

# Parameters
N = 300           # number of radial samples in real space
M = 300           # number of samples in Fourier (rho) space
r_max = 16000/0.22      # max radius in real space
rho_max = 20*np.pi/180/3600     # max frequency in Fourier space

# Grids
r = np.linspace(0, r_max, N)
rho = np.linspace(0, rho_max, M)

# Initialize profiles
f = np.zeros(N)
F = np.zeros(M)

# Hankel transform (zeroth-order)
def hankel_transform(f_real):
    # integrand: shape (N, M)
    R, P = np.meshgrid(r, rho, indexing='ij')
    integrand = f_real[:, None] * j0(2 * np.pi * R * P) * R
    # integrate over r-axis
    F =  2 * np.pi * np.trapz(integrand, r, axis=0)
    return F / F.max()

# Inverse Hankel transform
def inverse_hankel_transform(F_fourier):
    R, P = np.meshgrid(r, rho, indexing='ij')
    integrand = F_fourier[None, :] * j0(2 * np.pi * R * P) * P
    # integrate over rho-axis
    f =  np.trapz(integrand, rho, axis=1)
    return f / f.max()

# Set up figure and two axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
line1, = ax1.plot(r, f, lw=2)
line2, = ax2.plot(rho, F, lw=2)

ax1.set_title('Real-space radial distribution f(r)')
ax1.set_xlim(0, r_max)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('r')
ax1.set_ylabel('f(r)')

ax2.set_title('Fourier-space radial profile F(ρ)')
ax2.set_xlim(0, rho_max)
ax2.set_ylim(-0.1, 1.1)
# ax2.set_xlabel('ρ')
ax2.set_ylabel('F(ρ)')

# Interaction state
drawing = False
mode = None  # 'real' or 'fourier'

# Event handlers
def on_press(event):
    global drawing, mode
    if event.inaxes == ax1:
        drawing = True
        mode = 'real'
    elif event.inaxes == ax2:
        drawing = True
        mode = 'fourier'

def on_release(event):
    global drawing, mode
    drawing = False
    mode = None

def on_motion(event):
    if not drawing or event.xdata is None or event.ydata is None:
        return
    if mode == 'real' and event.inaxes == ax1:
        # find nearest index in r
        idx = np.argmin(np.abs(r - event.xdata))
        # update real-space profile
        f[idx] = event.ydata
        # recompute Fourier
        F[:] = hankel_transform(f)
    elif mode == 'fourier' and event.inaxes == ax2:
        # find nearest index in rho
        idx = np.argmin(np.abs(rho - event.xdata))
        # update Fourier-space profile
        F[idx] = event.ydata
        # recompute real-space
        f[:] = inverse_hankel_transform(F)
    else:
        return
    # update plots
    line1.set_ydata(f)
    line2.set_ydata(F)
    fig.canvas.draw_idle()

if __name__ == '__main__':

    # Connect events
    gfig = fig.canvas
    gfig.mpl_connect('button_press_event', on_press)
    gfig.mpl_connect('button_release_event', on_release)
    gfig.mpl_connect('motion_notify_event', on_motion)

    plt.tight_layout()
    plt.show()
