import numpy as np 
import matplotlib.pyplot as plt
import scipy

vs1 = np.loadtxt("signal1.txt")
vs2 = np.loadtxt("signal2.txt")
ts = np.arange(vs1.shape[0])

# plt.plot(ts,vs1)
# plt.plot(ts,vs2)
# plt.show()


# hs1 = np.loadtxt("hilbert1.txt")
# hs2 = np.loadtxt("hilbert2.txt")
# ts = np.arange(hs1.shape[0])

# plt.plot(ts,hs1)
# plt.plot(ts,hs2)
# plt.plot(ts,hs2-hs1)
# plt.show()


# phase = np.loadtxt("phase.txt")
# ts = np.arange(phase.shape[0])
# plt.plot(ts,phase)
# plt.show()

# print(np.mean(phase))


print(scipy.signal.hilbert(v1)