from __future__ import division
# __author__ = 'WeiFu'
import pdb
from matplotlib import pyplot as plt
import numpy as np

def printdic(out):
  for key, val in out.iteritems():
    print key + " " + str(val) + "\n"


def getData(src="./result/0929/epv=10*2_with_scipy.txt"):
  src="./result/20160806/20160806_Small_N.txt"
  def toFloat(x):
    try:
      return float(x)
    except ValueError, e:
      return x

  out = {}
  f = open(src, "r")
  content = f.read().splitlines()
  names = ["HDP-Scipy","N-50","N-100","N-150","N-200"]
  names_index = [ k for k, i in enumerate(content[0].split("|")) if i.strip() in names]
  for row in content[1:]:
    cell = [toFloat(i.strip()) for i in row.split("|")]
    out[cell[0]] = [cell[i] for i in names_index]
  # printdic(out)
  return out


def run(N=4):
  data = getData()
  label = {'EQ':'a','JDT':'b','LC':'c','ML':'d','PDE':'e','apache':'f','safe':'g','zxing':'h',
           'ant-1.3':'i','arc':'j','camel-1.0':'k','poi-1.5':'l','redaktor':'m','skarbonka':'n',
           'tomcat':'o','velocity-1.4':'p','xalan-2.4':'q','xerces-1.2':'r','CM1':'s','MW1':'t',
           'PC1':'u','PC3':'v','PC4':'w','JM1':'x','PC2':'y','PC5':'z','MC1':'A','MC2':'B','KC3':'C','ar1':'D','ar3':'E','ar4':'F','ar5':'G','ar6':'H',}
  samplesize = [50, 100, 150, 200]
  x = np.linspace(0, 33, 34)
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
    label_sorted = [ label[i[0]] for i in onePlot_sorted]
    print [i[0] for i in onePlot_sorted]
    print label_sorted
    for i,z in enumerate(x[:]):
      y = 0.8 if min(base[i],proposed[i])<0.5 else  min(base[i],proposed[i])
      ax_lst[num].annotate(label_sorted[i], xy = (z,y-0.15), textcoords = "data")

    ax_lst[num].plot(x, base, 'ko-', label='All Data')
    ax_lst[num].plot(x, proposed, 'ro-', label='Sampled with N= ' + str(samplesize[num]), color=cols[num])
    ax_lst[num].legend(fontsize='small',loc = 0)
    ax_lst[num].set_xlim([-1, 34])
    ax_lst[num].set_ylim([0, 1])
    ax_lst[num].set_yticks(np.arange(0.3, 0.9, 0.3))
  f.subplots_adjust(wspace=0, hspace=0)
  f.text(0.04, 0.5, 'AUC values', va='center', rotation='vertical', fontsize=11)
  plt.xlabel('Data sets, sorted by improvements')
  plt.savefig('./result/20160806/Small_N.eps', format='eps',dpi= 1000)
  plt.show()


if __name__ == "__main__":
  run()
