
def reduce_duplicated_hyph(aname):
    aname2 = []
    prev = aname[0]
    for c in aname[1:]:
        if c == '-':
            if prev == '-':
                pass
            else:
                aname2.append(prev)
        else:
            aname2.append(prev)
        prev = c
    aname2.append(c)
    aname2 = ''.join(i for i in aname2)
    return aname2