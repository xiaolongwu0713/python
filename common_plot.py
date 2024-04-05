import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# pre-defined colors
#colors=['orangered','skyblue','orange','springgreen','aquamarine','darkgoldrod','royalblue']

# default colors sequence
#import matplotlib.pyplot as plt
#prop_cycle = plt.rcParams['axes.prop_cycle']
#colors = prop_cycle.by_key()['color']
# or get the default color sequence below:
color_names=list(mcolors.TABLEAU_COLORS.keys())
color_codes=list(mcolors.TABLEAU_COLORS.values())

def barplot_annotate_brackets(layer,num1, num2, data, center, height, yerr=None, dh=.05, barh=.02, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.
    :param layer: 2nd layer bracket need on top on first layer bracket.
    :param num1: 1st bar
    :param num2: 2nd bar
    :param data: p-value or other string
    :param center: bar location
    :param height: mean
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: relative position of braket tips
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.
            if maxasterix and len(text) == maxasterix:
                break
        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]


    #if yerr:
    if ly<0 and ry<0:
        ly -= yerr[num1]
        ry -= yerr[num2]
    elif ly>0 and ry>0:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    barx = [lx, lx, rx, rx]
    if ly < 0 and ry < 0:
        y = min(ly, ry) - dh
        if layer!=0:
            bary = [y, y - barh, y - barh, y]
            bary = [i - barh * layer - 2*barh for i in bary]
        elif layer==0:
            bary = [y, y - barh, y - barh, y]
        text_y = min(bary)
        # bary = [i - barh * layer - 0.5 for i in bary]
        mid = ((lx + rx) / 2, text_y - 2 * barh)
    elif ly > 0 and ry > 0:
        y = max(ly, ry) + dh
        if layer!=0:
            bary = [y, y + barh, y + barh, y]
            bary=[i + barh * layer + 2*barh for i in bary]
        elif layer==0:
            bary = [y, y + barh, y + barh, y]
        text_y = max(bary)
        #bary = [i + barh * layer + 0.5 for i in bary]
        mid = ((lx + rx) / 2, text_y)
    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)
