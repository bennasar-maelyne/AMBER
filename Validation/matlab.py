from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact

#Choose the patient
dir="./Patient_1"

data = loadmat(f"{dir}/b0all_qb32_D19.mat")

print(data.keys())
#print(data["__header__"])
#print(data["__version__"])
#print(data["__globals__"])
#print(data["b0all"])

signal=data["b0all"]

print(f"Array shape is {signal.shape}")
print(f"The max of signal values is {np.max(signal)} at voxel {np.where(signal==np.max(signal))}")
print(f"Number of non-zero values is : {np.count_nonzero(signal)}")

signal_t=np.transpose(signal, (0,2,1,3))
slice=signal_t[:,:,80,1]

plt.imshow(slice,cmap="grey")
plt.colorbar()
plt.title("Slice ")
plt.axis('off')
plt.show()

volume=signal_t[:,:,:,0]

def plot_slice(z):
    plt.figure(figsize=(8, 8))
    plt.imshow(volume[:, :, z], cmap='gray')
    plt.title(f"Slice {z}")
    plt.axis('off')
    plt.show()

interact(plot_slice, z=(0, volume.shape[0] - 1, 1))