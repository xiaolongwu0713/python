from gesture.config import *
from scipy import signal
import mne
# save list to numpy
#np.save(data_dir+'adhoc/dataset_test.npy', np.array(test_epochs, dtype=object), allow_pickle=True)
# load list
data = np.load(data_dir+'adhoc/dataset_test.npy', allow_pickle=True)

fig,ax=plt.subplots()
a=data[0][0,2,2600:3600] #(2, 208, 4001)
b=data[0][1,2,3000:4000] #(2, 208, 4001)
reference=data[1][0,2,3000:4000]
ax.plot(a)
ax.plot(b)
ax.plot(reference)

noise = np.random.normal(0,0.2,len(b))
b=b+noise
ax.plot(b+noise)
ax.clear()

filename=data_dir+'adhoc/good_signal.pdf'
fig.savefig(filename)
filename=data_dir+'adhoc/bad_signal.pdf'
fig.savefig(filename)
filename=data_dir+'adhoc/reference_signal.pdf'
fig.savefig(filename)

f,psd=signal.welch(a,1000,'hamm',256,scaling='spectrum')
ax.semilogy(f,np.absolute(psd))
filename=data_dir+'adhoc/good_signal_PSD.pdf'
fig.savefig(filename)

f,psd=signal.welch(b,1000,'hamm',256,scaling='spectrum')
ax.semilogy(f,np.absolute(psd))
filename=data_dir+'adhoc/bad_signal_PSD.pdf'
fig.savefig(filename)

f,psd=signal.welch(reference,1000,'hamm',256,scaling='spectrum')
ax.semilogy(f,np.absolute(psd))
filename=data_dir+'adhoc/reference_signal_PSD.pdf'
fig.savefig(filename)

