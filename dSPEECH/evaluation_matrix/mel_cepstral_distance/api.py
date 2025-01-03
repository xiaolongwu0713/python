from logging import getLogger
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

from dSPEECH.evaluation_matrix.mel_cepstral_distance.alignment import align_MC_s_D, align_X_km, align_X_kn
from dSPEECH.evaluation_matrix.mel_cepstral_distance.computation import (get_average_MCD, get_MC_X_ik, get_MCD_k, get_w_n_m,
                                               get_X_km, get_X_kn)
from dSPEECH.evaluation_matrix.mel_cepstral_distance.helper import (get_n_fft_bins, ms_to_samples, norm_audio_signal,
                                          resample_if_necessary)
from dSPEECH.evaluation_matrix.mel_cepstral_distance.silence import (remove_silence_MC_X_ik, remove_silence_rms,
                                           remove_silence_X_km, remove_silence_X_kn)


def get_amplitude_spectrogram(audio: Path, *, sample_rate: Optional[int] = None, n_fft: float = 32, win_len: float = 32, hop_len: float = 8, window: Literal["hamming", "hanning"] = "hanning", norm_audio: bool = True, remove_silence: bool = False, silence_threshold: Optional[float] = None) -> npt.NDArray[np.complex128]:
  if sample_rate is not None and not 0 < sample_rate:
    raise ValueError("sample_rate must be > 0")

  if not n_fft > 0:
    raise ValueError("n_fft must be > 0")

  if not 0 < win_len:
    raise ValueError("win_len must be > 0")

  if not 0 < hop_len:
    raise ValueError("hop_len must be > 0")

  if window not in ["hamming", "hanning"]:
    raise ValueError("window must be 'hamming' or 'hanning")

  sr, signal = wavfile.read(audio)

  if sample_rate is None:
    sample_rate = sr

  n_fft_samples = ms_to_samples(n_fft, sample_rate)

  if len(signal) == 0:
    logger = getLogger(__name__)
    logger.warning("audio is empty")
    empty_spec = np.empty(
      (0, get_n_fft_bins(n_fft_samples)),
      dtype=np.complex128
    )
    return empty_spec

  signal = resample_if_necessary(signal, sr, sample_rate)

  n_fft_is_two_power = n_fft_samples & (n_fft_samples - 1) == 0

  if not n_fft_is_two_power:
    logger = getLogger(__name__)
    logger.warning(
      f"n_fft ({n_fft}ms / {n_fft_samples} samples) should be a power of 2 in samples for faster computation")

  if n_fft != win_len:
    logger = getLogger(__name__)
    logger.warning(f"n_fft ({n_fft}ms) should be equal to win_len ({win_len}ms)")
    if n_fft < win_len:
      logger.warning(f"truncating windows to n_fft ({n_fft}ms)")
    else:
      assert n_fft > win_len
      logger.warning(f"padding windows to n_fft ({n_fft}ms)")

  if norm_audio:
    signal = norm_audio_signal(signal)

  win_len_samples = ms_to_samples(win_len, sample_rate)

  if remove_silence:
    if silence_threshold is None:
      raise ValueError("silence_threshold must be set")

    if not 0 <= silence_threshold:
      raise ValueError("silence_threshold must be greater than or equal to 0 RMS")

    signal = remove_silence_rms(
      signal, silence_threshold,
      min_silence_samples=win_len_samples
    )

    if len(signal) == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, audio is empty")
      empty_spec = np.empty(
        (0, get_n_fft_bins(n_fft_samples)),
        dtype=np.complex128
      )
      return empty_spec

  # STFT - Shape: (#Frames, Bins)
  hop_len_samples = ms_to_samples(hop_len, sample_rate)
  X_km_A = get_X_km(signal, n_fft_samples, win_len_samples, hop_len_samples, window)
  return X_km_A


def get_mel_spectrogram(amp_spec: npt.NDArray[np.complex128], sample_rate: int, n_fft: float, /, *, M: int = 20, fmin: int = 0, fmax: Optional[int] = None, remove_silence: bool = False, silence_threshold: Optional[float] = None) -> npt.NDArray:
  # amp_spec = X_km

  if len(amp_spec.shape) != 2:
    raise ValueError(f"amplitude spectrogram must have 2 dimensions but got {len(amp_spec.shape)}")

  if not M > 0:
    raise ValueError("M must be > 0")

  if amp_spec.shape[0] == 0:
    logger = getLogger(__name__)
    logger.warning("spectrogram is empty")
    empty_mel_spec = np.empty((0, M), dtype=np.float64)
    return empty_mel_spec

  if amp_spec.shape[1] == 0:
    raise ValueError("spectrogram must have at least 1 frequency bin")

  if not 0 < n_fft:
    raise ValueError("n_fft must be > 0")

  if not 0 < sample_rate:
    raise ValueError("sample_rate must be > 0")

  if fmax is not None:
    if not 0 < fmax <= sample_rate // 2:
      raise ValueError(f"fmax must be in (0, sample_rate // 2], i.e., (0, {sample_rate//2}]")
  else:
    fmax = sample_rate // 2

  if not 0 <= fmin < fmax:
    raise ValueError(f"fmin must be in [0, fmax), i.e., [0, {fmax})")

  n_fft_samples = ms_to_samples(n_fft, sample_rate)
  if get_n_fft_bins(n_fft_samples) != amp_spec.shape[1]:
    raise ValueError(
      f"n_fft (in samples) // 2 + 1 must match the number of frequency bins in the spectrogram but got {n_fft_samples // 2 + 1} != {amp_spec.shape[1]}")

  if remove_silence:
    if silence_threshold is None:
      raise ValueError("silence_threshold must be set")

    amp_spec = remove_silence_X_km(amp_spec, silence_threshold)

    if amp_spec.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, spectrogram is empty")
      empty_mel_spec = np.empty((0, M), dtype=np.float64)
      return empty_mel_spec

  w_n_m = get_w_n_m(sample_rate, n_fft_samples, M, fmin, fmax)
  X_kn = get_X_kn(amp_spec, w_n_m)
  return X_kn


def get_mfccs(mel_spec: npt.NDArray, /, *, remove_silence: bool = False, silence_threshold: Optional[float] = None) -> npt.NDArray:
  if len(mel_spec.shape) != 2:
    raise ValueError(f"mel-spectrogram must have 2 dimensions but got {len(mel_spec.shape)}")

  if mel_spec.shape[1] == 0:
    raise ValueError("mel-spectrogram must have at least 1 mel-band")

  if mel_spec.shape[0] == 0:
    logger = getLogger(__name__)
    logger.warning("mel-spectrogram is empty")
    empty_mfccs = np.empty((mel_spec.shape[1], 0), dtype=np.float64)
    return empty_mfccs

  if remove_silence:
    if silence_threshold is None:
      raise ValueError("silence_threshold must be set")

    mel_spec = remove_silence_X_kn(mel_spec, silence_threshold)

    if mel_spec.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, mel-spectrogram is empty")
      empty_mfccs = np.empty((mel_spec.shape[1], 0), dtype=np.float64)
      return empty_mfccs

  MC_X_ik = get_MC_X_ik(mel_spec, mel_spec.shape[1])
  return MC_X_ik


def compare_audio_files(audio_A: Path, audio_B: Path, /, *, sample_rate: Optional[int] = None,
                        n_fft: float = 32, win_len: float = 32, hop_len: float = 8, window: Literal["hamming", "hanning"] = "hanning",
                        fmin: int = 0, fmax: Optional[int] = None, M: int = 20, s: int = 1, D: int = 16,
                        aligning: Literal["pad", "dtw"] = "dtw", align_target: Literal["spec", "mel", "mfcc"] = "mel",
                        remove_silence: Literal["no", "sig", "spec", "mel", "mfcc"] = "no", silence_threshold_A: Optional[float] = None,
                        silence_threshold_B: Optional[float] = None, norm_audio: bool = True,
                        dtw_radius: Optional[int] = 10) -> Tuple[float, float]:
  """
  - silence is removed before alignment
  - high freq is max sr/2
  - n_fft should be equal to win_len
  - n_fft should be a power of 2 in samples
  - dtw_radius: 1 means fastest but less accurate, None means slowest but most accurate alignment, 10 is a good trade-off
  """
  if remove_silence not in ["no", "sig", "spec", "mel", "mfcc"]:
    raise ValueError("remove_silence must be 'no', 'sig', 'spec', 'mel' or 'mfcc'")

  if sample_rate is not None and not 0 < sample_rate:
    raise ValueError("sample_rate must be > 0")

  if not n_fft > 0:
    raise ValueError("n_fft must be > 0")

  if not 0 < win_len:
    raise ValueError("win_len must be > 0")

  if not 0 < hop_len:
    raise ValueError("hop_len must be > 0")

  if window not in ["hamming", "hanning"]:
    raise ValueError("window must be 'hamming' or 'hanning'")

  sr1, signalA = wavfile.read(audio_A)
  sr2, signalB = wavfile.read(audio_B)

  if signalA.dtype != signalB.dtype:
    logger = getLogger(__name__)
    logger.warning(f"audio A and B have different data types ({signalA.dtype} != {signalB.dtype})")

  if len(signalA) == 0:
    logger = getLogger(__name__)
    logger.warning("audio A is empty")
    return np.nan, np.nan

  if len(signalB) == 0:
    logger = getLogger(__name__)
    logger.warning("audio B is empty")
    return np.nan, np.nan

  if sample_rate is None:
    sample_rate = min(sr1, sr2)

  signalA = resample_if_necessary(signalA, sr1, sample_rate)
  signalB = resample_if_necessary(signalB, sr2, sample_rate)

  n_fft_samples = ms_to_samples(n_fft, sample_rate)
  n_fft_is_two_power = n_fft_samples & (n_fft_samples - 1) == 0

  if not n_fft_is_two_power:
    logger = getLogger(__name__)
    logger.warning(
      f"n_fft ({n_fft}ms / {n_fft_samples} samples) should be a power of 2 in samples for faster computation")

  if n_fft != win_len:
    logger = getLogger(__name__)
    logger.warning(f"n_fft ({n_fft}ms) should be equal to win_len ({win_len}ms)")
    if n_fft < win_len:
      logger.warning(f"truncating windows to n_fft ({n_fft}ms)")
    else:
      assert n_fft > win_len
      logger.warning(f"padding windows to n_fft ({n_fft}ms)")

  if norm_audio:
    signalA = norm_audio_signal(signalA)
    signalB = norm_audio_signal(signalB)

  win_len_samples = ms_to_samples(win_len, sample_rate)

  if remove_silence == "sig":
    if silence_threshold_A is None:
      raise ValueError("silence_threshold_A must be set")

    if silence_threshold_B is None:
      raise ValueError("silence_threshold_B must be set")

    if not 0 <= silence_threshold_A:
      raise ValueError("silence_threshold_A must be greater than or equal to 0 RMS")

    if not 0 <= silence_threshold_B:
      raise ValueError("silence_threshold_B must be greater than or equal to 0 RMS")

    signalA = remove_silence_rms(
      signalA, silence_threshold_A,
      min_silence_samples=win_len_samples
    )

    signalB = remove_silence_rms(
      signalB, silence_threshold_B,
      min_silence_samples=win_len_samples
    )

    if len(signalA) == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, audio A is empty")
      return np.nan, np.nan

    if len(signalB) == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, audio B is empty")
      return np.nan, np.nan

    remove_silence = "no"

  # STFT - Shape: (#Frames, Bins)
  hop_len_samples = ms_to_samples(hop_len, sample_rate)
  X_km_A = get_X_km(signalA, n_fft_samples, win_len_samples, hop_len_samples, window)
  X_km_B = get_X_km(signalB, n_fft_samples, win_len_samples, hop_len_samples, window)

  mean_mcd_over_all_k, res_penalty = compare_amplitude_spectrograms(
    X_km_A, X_km_B, sample_rate, n_fft, fmin=fmin, fmax=fmax, M=M, s=s, D=D, aligning=aligning, align_target=align_target, remove_silence=remove_silence, silence_threshold_A=silence_threshold_A, silence_threshold_B=silence_threshold_B, dtw_radius=dtw_radius
  )

  return mean_mcd_over_all_k, res_penalty


def compare_amplitude_spectrograms(amp_spec_A: npt.NDArray[np.complex128], amp_spec_B: npt.NDArray[np.complex128],
                                   sample_rate: int, n_fft: float, /, *, fmin: int = 0, fmax: Optional[int] = None,
                                   M: int = 20, s: int = 1, D: int = 16, aligning: Literal["pad", "dtw"] = "dtw",
                                   align_target: Literal["spec", "mel", "mfcc"] = "spec",
                                   remove_silence: Literal["no", "spec", "mel", "mfcc"] = "no",
                                   silence_threshold_A: Optional[float] = None, silence_threshold_B: Optional[float] = None,
                                   dtw_radius: Optional[int] = 10) -> Tuple[float, float]:
  if len(amp_spec_A.shape) != 2:
    raise ValueError(
      f"amplitude spectrogram A must have 2 dimensions but got {len(amp_spec_A.shape)}")

  if len(amp_spec_B.shape) != 2:
    raise ValueError(
      f"amplitude spectrogram B must have 2 dimensions but got {len(amp_spec_B.shape)}")

  if amp_spec_A.shape[0] == 0:
    logger = getLogger(__name__)
    logger.warning("spectrogram A is empty")
    return np.nan, np.nan

  if amp_spec_B.shape[0] == 0:
    logger = getLogger(__name__)
    logger.warning("spectrogram B is empty")
    return np.nan, np.nan

  if not amp_spec_A.shape[1] == amp_spec_B.shape[1]:
    raise ValueError(
      f"both spectrograms must have the same number of frequency bins but got {amp_spec_A.shape[1]} != {amp_spec_B.shape[1]}")

  assert amp_spec_A.shape[1] == amp_spec_B.shape[1]
  n_fft_bins = amp_spec_A.shape[1]

  if n_fft_bins == 0:
    raise ValueError("spectrograms must have at least 1 frequency bin")

  n_fft_samples = ms_to_samples(n_fft, sample_rate)
  if get_n_fft_bins(n_fft_samples) != n_fft_bins:
    raise ValueError(
      f"n_fft (in samples) // 2 + 1 must match the number of frequency bins in the spectrogram but got {n_fft_samples // 2 + 1} != {n_fft_bins}")
  assert n_fft_samples > 0
  assert sample_rate > 0

  if aligning not in ["pad", "dtw"]:
    raise ValueError("aligning must be 'pad' or 'dtw'")

  if remove_silence not in ["no", "spec", "mel", "mfcc"]:
    raise ValueError("remove_silence must be 'no', 'spec', 'mel' or 'mfcc'")

  if align_target == "spec":
    if remove_silence == "mel":
      raise ValueError(
        "cannot remove silence from mel-spectrogram after both spectrograms were aligned")
    if remove_silence == "mfcc":
      raise ValueError(
        "cannot remove silence from MFCCs after both spectrograms were aligned")

  if fmax is not None:
    if not 0 < fmax <= sample_rate // 2:
      raise ValueError(f"fmax must be in (0, sample_rate // 2], i.e., (0, {sample_rate//2}]")
  else:
    fmax = sample_rate // 2

  if not 0 <= fmin < fmax:
    raise ValueError(f"fmin must be in [0, fmax), i.e., [0, {fmax})")

  if not M > 0:
    raise ValueError("M must be > 0")

  if remove_silence == "spec":
    if silence_threshold_A is None:
      raise ValueError("silence_threshold_A must be set")
    if silence_threshold_B is None:
      raise ValueError("silence_threshold_B must be set")

    amp_spec_A = remove_silence_X_km(amp_spec_A, silence_threshold_A)
    amp_spec_B = remove_silence_X_km(amp_spec_B, silence_threshold_B)

    if amp_spec_A.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, spectrogram A is empty")
      return np.nan, np.nan

    if amp_spec_B.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, spectrogram B is empty")
      return np.nan, np.nan

    remove_silence = "no"

  penalty: float
  aligned_here: bool = False
  if align_target == "spec":
    if aligning == "dtw" and dtw_radius is not None and not 1 <= dtw_radius:
      raise ValueError("dtw_radius must be None or greater than or equal to 1")
    amp_spec_A, amp_spec_B, penalty = align_X_km(amp_spec_A, amp_spec_B, aligning, dtw_radius)
    aligned_here = True
    align_target = "mel"
    aligning = "pad"

  # Mel-Bank - Shape: (N, #Frames)
  w_n_m = get_w_n_m(sample_rate, n_fft_samples, M, fmin, fmax)

  # Mel-Spectrogram - Shape: (#Frames, #N)
  X_kn_A = get_X_kn(amp_spec_A, w_n_m)
  X_kn_B = get_X_kn(amp_spec_B, w_n_m)

  mean_mcd_over_all_k, res_penalty = compare_mel_spectrograms(
    X_kn_A, X_kn_B,
    s=s, D=D, aligning=aligning, align_target=align_target, remove_silence=remove_silence, silence_threshold_A=silence_threshold_A, silence_threshold_B=silence_threshold_B,
    dtw_radius=dtw_radius
  )

  if aligned_here:
    assert res_penalty == 0
  else:
    assert "penalty" not in locals()
    assert res_penalty is not None
    penalty = res_penalty

  return mean_mcd_over_all_k, penalty


def compare_mel_spectrograms(mel_spec_A: npt.NDArray, mel_spec_B: npt.NDArray, /, *, s: int = 1, D: int = 16,
                             aligning: Literal["pad", "dtw"] = "dtw", align_target: Literal["mel", "mfcc"] = "mel",
                             remove_silence: Literal["no", "mel", "mfcc"] = "no", silence_threshold_A: Optional[float] = None,
                             silence_threshold_B: Optional[float] = None, dtw_radius: Optional[int] = 10) -> Tuple[float, float]:
  if not len(mel_spec_A.shape) == 2:
    raise ValueError(f"mel-spectrogram A must have 2 dimensions but got {len(mel_spec_A.shape)}")

  if not len(mel_spec_B.shape) == 2:
    raise ValueError(f"mel-spectrogram B must have 2 dimensions but got {len(mel_spec_B.shape)}")

  if len(mel_spec_A) == 0:
    logger = getLogger(__name__)
    logger.warning("mel-spectrogram A is empty")
    return np.nan, np.nan

  if len(mel_spec_B) == 0:
    logger = getLogger(__name__)
    logger.warning("mel-spectrogram B is empty")
    return np.nan, np.nan

  if not mel_spec_A.shape[1] == mel_spec_B.shape[1]:
    raise ValueError("both mel-spectrograms must have the same number of mel-bands")
  M = mel_spec_A.shape[1]

  if not M > 0:
    raise ValueError("mel-spectrograms must have at least 1 mel-band")

  if aligning not in ["pad", "dtw"]:
    raise ValueError("aligning must be 'pad' or 'dtw'")

  if remove_silence not in ["no", "mel", "mfcc"]:
    raise ValueError("remove_silence must be 'no', 'mel' or 'mfcc'")

  if align_target not in ["mel", "mfcc"]:
    raise ValueError("align_target must be 'mel' or 'mfcc'")

  if align_target == "mel" and remove_silence == "mfcc":
    raise ValueError(
        "cannot remove silence from MFCCs after both mel-spectrograms were aligned")

  if D > M:
    raise ValueError(f"D must be <= number of mel-bands ({M})")

  if remove_silence == "mel":
    if silence_threshold_A is None:
      raise ValueError("silence_threshold_A must be set")
    if silence_threshold_B is None:
      raise ValueError("silence_threshold_B must be set")

    mel_spec_A = remove_silence_X_kn(mel_spec_A, silence_threshold_A)
    mel_spec_B = remove_silence_X_kn(mel_spec_B, silence_threshold_B)

    if mel_spec_A.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, mel-spectrogram A is empty")
      return np.nan, np.nan

    if mel_spec_B.shape[0] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, mel-spectrogram B is empty")
      return np.nan, np.nan

    remove_silence = "no"

  penalty: float
  aligned_here: bool = False
  if align_target == "mel":
    if aligning == "dtw" and dtw_radius is not None and not 1 <= dtw_radius:
      raise ValueError("dtw_radius must be None or greater than or equal to 1")
    mel_spec_A, mel_spec_B, penalty = align_X_kn(mel_spec_A, mel_spec_B, aligning, dtw_radius) #
    aligned_here = True
    align_target = "mfcc"
    aligning = "pad"

  # Shape: (N, #Frames)
  MC_X_ik = get_MC_X_ik(mel_spec_A, M)
  MC_Y_ik = get_MC_X_ik(mel_spec_B, M)

  remove_silence_mfcc = remove_silence == "mfcc"

  mean_mcd_over_all_k, res_penalty = compare_mfccs(
    MC_X_ik, MC_Y_ik,
    s=s, D=D, aligning=aligning, remove_silence=remove_silence_mfcc, silence_threshold_A=silence_threshold_A, silence_threshold_B=silence_threshold_B, dtw_radius=dtw_radius
  )

  if aligned_here:
    assert res_penalty == 0
  else:
    assert "penalty" not in locals()
    assert res_penalty is not None
    penalty = res_penalty

  return mean_mcd_over_all_k, penalty


def compare_mfccs(mfccs_A: npt.NDArray, mfccs_B: npt.NDArray, /, *, s: int = 1, D: int = 16, aligning: Literal["pad", "dtw"] = "dtw", remove_silence: bool = False, silence_threshold_A: Optional[float] = None, silence_threshold_B: Optional[float] = None, dtw_radius: Optional[int] = 10) -> Tuple[float, float]:
  if not len(mfccs_A.shape) == 2:
    raise ValueError(f"MFCCs A must have 2 dimensions but got {len(mfccs_A.shape)}")

  if not len(mfccs_B.shape) == 2:
    raise ValueError(f"MFCCs B must have 2 dimensions but got {len(mfccs_B.shape)}")

  if mfccs_A.shape[1] == 0:
    logger = getLogger(__name__)
    logger.warning("MFCCs A are empty")
    return np.nan, np.nan

  if mfccs_B.shape[1] == 0:
    logger = getLogger(__name__)
    logger.warning("MFCCs B are empty")
    return np.nan, np.nan

  if not mfccs_A.shape[0] == mfccs_B.shape[0]:
    raise ValueError("both MFCCs must have the same number of coefficients")

  M = mfccs_A.shape[0]

  if not M > 0:
    raise ValueError("MFCCs must have at least 1 coefficient")

  if not D <= M:
    raise ValueError(f"D must be <= number of MFCC coefficients ({M})")

  if not 0 <= s < D:
    raise ValueError("s must be in [0, D)")

  assert D >= 1

  if aligning not in ["pad", "dtw"]:
    raise ValueError("aligning must be 'pad' or 'dtw'")

  if remove_silence:
    if silence_threshold_A is None:
      raise ValueError("silence_threshold_A must be set")
    if silence_threshold_B is None:
      raise ValueError("silence_threshold_B must be set")

    mfccs_A = remove_silence_MC_X_ik(mfccs_A, silence_threshold_A)
    mfccs_B = remove_silence_MC_X_ik(mfccs_B, silence_threshold_B)

    if mfccs_A.shape[1] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, MFCCs A are empty")
      return np.nan, np.nan

    if mfccs_B.shape[1] == 0:
      logger = getLogger(__name__)
      logger.warning("after removing silence, MFCCs B are empty")
      return np.nan, np.nan

  if aligning == "dtw" and dtw_radius is not None and not 1 <= dtw_radius:
    raise ValueError("dtw_radius must be None or greater than or equal to 1")

  mfccs_A, mfccs_B, penalty = align_MC_s_D(mfccs_A, mfccs_B, s, D, aligning, dtw_radius)

  MCD_k = get_MCD_k(mfccs_A, mfccs_B, s, D)
  mean_mcd_over_all_k = get_average_MCD(MCD_k)

  return mean_mcd_over_all_k#, penalty
