from __future__ import division
# __author__ = 'WeiFu'
from matplotlib import pyplot as plt
import numpy as np


def printdic(out):
  for key, val in out.iteritems():
    print key + " " + str(val) + "\n"


def getData(src="./result/0924/epv=10_2_0924_with_SCIPY.txt"):
  def toFloat(x):
    try:
      return float(x)
    except ValueError, e:
      return x

  out = {}
  f = open(src, "r")
  content = f.read().splitlines()
  for row in content[1:]:
    cell = [toFloat(i.strip()) for i in row.split("|")]
    out[cell[0]] = [cell[2], cell[3], cell[5], cell[7], cell[9]]
  # printdic(out)
  return out


def run(N=4):
  data = getData()
  samplesize = [50, 100, 150, 200]
  x = np.linspace(0, 27, 28)
  f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
  ax_lst = [ax1, ax2, ax3, ax4]
  cols = ['r', 'y', 'g', 'b']
  for num in range(N):
    onePlot = {}
    for key, val in data.iteritems():
      onePlot[key] = [val[0], val[num + 1], val[num + 1] - val[0]]
    onePlot_sorted = sorted(onePlot.items(), key=lambda x: x[-1][-1])
    base = [i[-1][0] for i in onePlot_sorted]
    proposed = [i[-1][1] for i in onePlot_sorted]

    ax_lst[num].plot(x, base, 'ko-', label='All Data')
    ax_lst[num].plot(x, proposed, 'ro-', label='Sampled with N= ' + str(samplesize[num]), color=cols[num])
    ax_lst[num].legend(fontsize='small',loc = 0)
    ax_lst[num].set_xlim([0, 28])
    ax_lst[num].set_ylim([0, 1])
    ax_lst[num].set_yticks(np.arange(0.3, 0.9, 0.3))
  f.subplots_adjust(wspace=0, hspace=0)
  f.text(0.04, 0.5, 'Improvements', va='center', rotation='vertical', fontsize=11)
  plt.xlabel('Data sets, sorted by improvements')
  plt.show()


if __name__ == "__main__":
  run()
