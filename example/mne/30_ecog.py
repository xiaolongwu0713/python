"""
.. _tut-working-with-ecog:

======================
Working with ECoG data
======================

MNE supports working with more than just MEG and EEG data. Here we show some
of the functions that can be used to facilitate working with
electrocorticography (ECoG) data.

This example shows how to use:

- ECoG data (`available here <https://openneuro.org/datasets/ds003029>`__) from
  an epilepsy patient during a seizure
- channel locations in FreeSurfer's ``fsaverage`` MRI space
- projection onto a pial surface

For a complementary example that involves sEEG data, channel locations in
MNI space, or projection into a volume, see :ref:`tut-working-with-seeg`.
"""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Chris Holdgraf <choldgraf@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#          Liberty Hamilton <libertyhamilton@gmail.com>
#
# License: BSD-3-Clause

# %%

import os.path as op

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mne_bids import BIDSPath, read_raw_bids
import matplotlib.pyplot as plt
plt.plot()
import mne
from mne.viz import plot_alignment, snapshot_brain_montage

print(__doc__)

# paths to mne datasets - sample ECoG and FreeSurfer subject
bids_root = mne.datasets.epilepsy_ecog.data_path()
sample_path = mne.datasets.sample.data_path()
subjects_dir = op.join(sample_path, 'subjects')

bids_path = BIDSPath(root=bids_root, subject='pt1', session='presurgery',task='ictal', datatype='ieeg', extension='vhdr')

raw = read_raw_bids(bids_path=bids_path, verbose=False)
raw.pick_types(ecog=True)
raw.load_data()
raw.notch_filter([60], trans_bandwidth=3)
raw.drop_channels(raw.info['bads'])
montage = raw.get_montage()
print(montage.get_positions()['coord_frame'])
montage.add_mni_fiducials(subjects_dir)
raw.set_montage(montage)
events, event_id = mne.events_from_annotations(raw)
epoch_length = 25  # seconds
epochs = mne.Epochs(raw, events, event_id=event_id['onset'], tmin=13, tmax=13 + epoch_length, baseline=None)
evoked = epochs.average().resample(200)
del epochs

fig = plot_alignment(raw.info, trans='fsaverage',subject='fsaverage', subjects_dir=subjects_dir,surfaces=['pial'], coord_frame='head')
mne.viz.set_3d_view(fig, azimuth=0, elevation=70)

xy, im = snapshot_brain_montage(fig, raw.info)

# %%
# Compute frequency features of the data
# --------------------------------------
#
# Next, we'll compute the signal power in the gamma (30-90 Hz) band,
# downsampling the result to 10 Hz (to save time).

sfreq = 10
gamma_power_t = evoked.copy().filter(30, 90).apply_hilbert(
    envelope=True).resample(sfreq)
gamma_info = gamma_power_t.info

# %%
# Visualize the time-evolution of the gamma power on the brain
# ------------------------------------------------------------
#
# Say we want to visualize the evolution of the power in the gamma band,
# instead of just plotting the average. We can use
# `matplotlib.animation.FuncAnimation` to create an animation and apply this
# to the brain figure.

# convert from a dictionary to array to plot
xy_pts = np.vstack([xy[ch] for ch in raw.info['ch_names']])

# get a colormap to color nearby points similar colors
cmap = get_cmap('viridis')

# create the figure of the brain with the electrode positions
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_title('Gamma power over time', size='large')
ax.imshow(im)
ax.set_axis_off()

# normalize gamma power for plotting
gamma_power = -100 * gamma_power_t.data / gamma_power_t.data.max()
# add the time course overlaid on the positions
x_line = np.linspace(-0.025 * im.shape[0], 0.025 * im.shape[0],
                     gamma_power_t.data.shape[1])
for i, pos in enumerate(xy_pts):
    x, y = pos
    color = cmap(i / xy_pts.shape[0])
    ax.plot(x_line + x, gamma_power[i] + y, linewidth=0.5, color=color)

# %%
# We can project gamma power from the sensor data to the nearest locations on
# the pial surface and visualize that:
#
# As shown in the plot, the epileptiform activity starts in the temporal lobe,
# progressing posteriorly. The seizure becomes generalized eventually, after
# this example short time section. This dataset is available using
# :func:`mne.datasets.epilepsy_ecog.data_path` for you to examine.

# sphinx_gallery_thumbnail_number = 3

xyz_pts = np.array([dig['r'] for dig in evoked.info['dig']])

src = mne.read_source_spaces(
    op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
stc = mne.stc_near_sensors(gamma_power_t, trans='fsaverage',
                           subject='fsaverage', src=src,
                           mode='nearest', subjects_dir=subjects_dir,
                           distance=0.02)
vmin, vmid, vmax = np.percentile(gamma_power_t.data, [10, 25, 90])
clim = dict(kind='value', lims=[vmin, vmid, vmax])
brain = stc.plot(surface='pial', hemi='rh', colormap='inferno', colorbar=False,
                 clim=clim, views=['lat', 'med'], subjects_dir=subjects_dir,
                 size=(250, 250), smoothing_steps='nearest',
                 time_viewer=False)
brain.add_sensors(raw.info, trans='fsaverage')

# You can save a movie like the one on our documentation website with:
# brain.save_movie(time_dilation=1, interpolation='linear', framerate=3,
#                  time_viewer=True)
