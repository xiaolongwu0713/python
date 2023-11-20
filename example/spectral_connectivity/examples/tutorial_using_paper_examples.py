import numpy as np
import matplotlib.pyplot as plt
from spectral_connectivity import Multitaper
from spectral_connectivity import Connectivity

#simulate_MVAR(coefficients, noise_covariance=noise_covariance, n_time_samples=n_time_samples,n_trials=500, n_burnin_samples=500)
def simulate_MVAR(coefficients, noise_covariance=None, n_time_samples=100,
                  n_trials=1, n_burnin_samples=100):
    '''
    Parameters
    ----------
    coefficients : array, shape (n_time_samples, n_lags, n_signals, n_signals)
    noise_covariance : array, shape (n_signals, n_signals)

    Returns
    -------
    time_series : array, shape (n_time_samples - n_burnin_samples,
                                n_trials, n_signals)

    '''
    n_lags, n_signals, _ = coefficients.shape # (1, 3, 3)
    if noise_covariance is None:
        noise_covariance = np.eye(n_signals)
    time_series = np.random.multivariate_normal( # (1500, 500, 3) (time_samples, trials, signal_channels)
        np.zeros((n_signals,)), noise_covariance,
        size=(n_time_samples + n_burnin_samples, n_trials))
    ## multiply the noise level
    noise = np.random.multivariate_normal(  # (1500, 500, 3) (time_samples, trials, signal_channels)
        np.zeros((n_signals,)), noise_covariance,
        size=(n_time_samples + n_burnin_samples, n_trials))
    noise_level=np.asarray([1,1,0.4,2,1,1,1.4])
    noise=noise*noise_level[np.newaxis,np.newaxis,:]

    for time_ind in np.arange(n_lags, n_time_samples + n_burnin_samples):
        for lag_ind in np.arange(n_lags):
            time_series[time_ind, ...] += np.matmul(
                coefficients[np.newaxis, np.newaxis, lag_ind, ...], # (1, 1, 3, 3)
                time_series[time_ind - (lag_ind + 1), ..., np.newaxis]).squeeze() # (500, 3, 1)
    time_series=time_series+noise
    return time_series[n_burnin_samples:, ...]

def simulate_MVAR2(coefficients, noise_covariance=None, n_trials=1,
                  n_burnin_samples=100, n_time_samples=500):
    """Simulate MVAR process
    Parameters
    ----------
    A : ndarray, shape (p, N, N)
        The AR coefficients where N is the number of signals
        and p the order of the model.
    n : int
        The number of time samples.
    sigma : array, shape (N,)
        The noise for each time series
    burnin : int
        The length of the burnin period (in samples).
    Returns
    -------
    X : ndarray, shape (N, n)
        The N time series of length n
    """
    n_lags, n_signals, _ = coefficients.shape
    A_2d = np.concatenate(coefficients, axis=1)
    time_series = np.zeros((n_time_samples + n_burnin_samples, n_signals))

    for time_ind in range(n_lags, n_time_samples + n_burnin_samples):
        noise = np.random.multivariate_normal(np.zeros((n_signals,)), noise_covariance)
        time_series[time_ind] = np.dot(
            A_2d,
            time_series[time_ind - n_lags:time_ind][::-1, :].ravel()) + noise

    return time_series[n_burnin_samples:]


def plot_directional(time_series, sampling_frequency, time_halfbandwidth_product=2):
    m = Multitaper(time_series,
                   sampling_frequency=sampling_frequency,
                   time_halfbandwidth_product=time_halfbandwidth_product,
                   start_time=0)
    c = Connectivity(fourier_coefficients=m.fft(),
                     frequencies=m.frequencies,
                     time=m.time)

    measures = dict(
        pairwise_spectral_granger=c.pairwise_spectral_granger_prediction(), # (1, 500, 3, 3)
        directed_transfer_function=c.directed_transfer_function(),
        partial_directed_coherence=c.partial_directed_coherence(),
        generalized_partial_directed_coherence=c.generalized_partial_directed_coherence(),
        #direct_directed_transfer_function=c.direct_directed_transfer_function(),
    )

    n_signals = time_series.shape[-1]
    signal_ind2, signal_ind1 = np.meshgrid(np.arange(n_signals), np.arange(n_signals))

    fig, axes = plt.subplots(n_signals, n_signals, figsize=(n_signals*1.5, n_signals*1.5), sharex=True)
    for ind1, ind2, ax in zip(signal_ind1.ravel(), signal_ind2.ravel(), axes.ravel()):
        for measure_name, measure in measures.items():
            ax.plot(c.frequencies, measure[0, :, ind1, ind2], label=measure_name,
                    linewidth=5, alpha=0.8)
        ax.set_title('x{} → x{}'.format(ind2 + 1, ind1 + 1), fontsize=15)
        ax.set_ylim((0, np.max([np.nanmax(np.stack(list(measures.values()))), 1.05])))

    axes[0, -1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.03)
    fig, axes = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
    axes.plot(c.frequencies, c.power().squeeze())
    plt.title('Power')

#def baccala_example2():
    '''Baccalá, L.A., and Sameshima, K. (2001). Partial directed coherence:
    a new concept in neural structure determination. Biological
    Cybernetics 84, 463–474.
    '''
sampling_frequency = 200
n_time_samples, n_lags, n_signals = 1000, 1, 3

coefficients = np.array(
    [[[ 0.5,  0.3,  0.4],
      [-0.5,  0.3,  1. ],
      [ 0. , -0.3, -0.2]]])

noise_covariance = np.eye(n_signals)

#return simulate_MVAR(coefficients, noise_covariance=noise_covariance, n_time_samples=n_time_samples,n_trials=500, n_burnin_samples=500),
# sampling_frequency
time_series=simulate_MVAR(coefficients, noise_covariance=noise_covariance,  # (1000, 500, 3) (time_stemp, trials, channels)
                      n_time_samples=n_time_samples,n_trials=500, n_burnin_samples=500)
sampling_frequency=200
#plot_directional(*baccala_example2(), time_halfbandwidth_product=1)
plot_directional(time_series, sampling_frequency, time_halfbandwidth_product=1)


#### example from paper: Vlachos, Ioannis; Krishnan, Balu; Treiman, David M.; Tsakalis, Konstantinos; Kugiumtzis, Dimitris; Iasemidis, Leon D. (2017): The Concept of Effective Inflow: Application to Interictal Localization of the Epileptogenic Focus From iEEG.
n_time_samples, n_lags, n_signals = 1000, 1, 7
coefficients = np.array(
    [[[0.5,0,0,0,0,0,0],
      [0.5,0,0,0,0,0,0],
      [0.5,0,0,0,0,0,0],
      [0.5,0,0,0,0,0,0],
      [0.5,0,0,0,0.3,0,0],
      [0.5,0,0,0,0,0.5,0],
      [0.5,0,0,0,0,0,0.3]
      ]])
noise_covariance = np.eye(n_signals)
sampling_frequency = 200
time_series=simulate_MVAR(coefficients, noise_covariance=noise_covariance,  # (1000, 500, 3) (time_stemp, trials, channels)
                      n_time_samples=n_time_samples,n_trials=500, n_burnin_samples=500)

plot_directional(time_series, sampling_frequency, time_halfbandwidth_product=1)























