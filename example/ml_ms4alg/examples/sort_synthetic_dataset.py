from ms4alg_pipelines import synthesize_sample_dataset
from ms4alg_pipelines import sort_dataset
import os
synthesize_sample_dataset(dataset_dir='dataset',opts={'verbose':'minimal'})

sort_dataset(dataset_dir='dataset',output_dir='output',adjacency_radius=-1,detect_threshold=3,opts={'verbose':'minimal'})

# Run this only if you are running jupyter lab locally and have ephys-viz installed:
os.system('ev-timeseries {} --firings={}'.format('output/filt.mda.prv','output/firings_curated.mda'))







