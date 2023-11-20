filename='G:/data/speech_RuiJin/P3/raw/EEG/profussion/save_data_from_curry.edf'

import mne
raw=mne.io.read_raw_edf(filename)

filename='G:/data/speech_RuiJin/P3/raw/EEG/profussion/save_data__from_curry.cdt'
raw=mne.io.read_raw_curry(filename)
raw.plot()


filename='G:/data/speech_RuiJin/P1/raw/EEG/curry/session2/ddd2 Data.cdt'
raw=mne.io.read_raw_curry(filename)


