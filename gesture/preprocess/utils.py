def getIndex(fMin, fMax, fstep, freq):
    freqs = [*range(fMin, fMax, fstep)]
    distance = [abs(fi - freq) for fi in freqs]
    index = distance.index(min(distance))
    return index