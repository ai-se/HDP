from __future__ import division
# __author__ = 'WeiFu'
import pdb
from matplotlib import pyplot as plt
import numpy as np

def printdic(out):
  for key, val in out.iteritems():
    print key + " " + str(val) + "\n"


def getData(src="./result/0929/epv=10*2_with_scipy.txt"):
  src="./result/1008/1008_random_N.txt"
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
           'tomcat':'o','velocity-1.4':'p','xalan-2.4':'q','xerces-1.2':'r','cm1':'s','mw1':'t',
           'pc1':'u','pc3':'v','pc4':'w','ar1':'x','ar3':'y','ar4':'z','ar5':'A','ar6':'B'}
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
    label_sorted = [ label[i[0]] for i in onePlot_sorted]
    print [i[0] for i in onePlot_sorted]
    print label_sorted
    for i,z in enumerate(x[:]):
      y = max(base[i],proposed[i])
      ax_lst[num].annotate(label_sorted[i], xy = (z,y+0.09), textcoords = "data")

    ax_lst[num].plot(x, base, 'ko-', label='All Data')
    ax_lst[num].plot(x, proposed, 'ro-', label='Sampled with N= ' + str(samplesize[num]), color=cols[num])
    ax_lst[num].legend(fontsize='small',loc = 0)
    ax_lst[num].set_xlim([-1, 28])
    ax_lst[num].set_ylim([0, 1])
    ax_lst[num].set_yticks(np.arange(0.3, 0.9, 0.3))
  f.subplots_adjust(wspace=0, hspace=0)
  f.text(0.04, 0.5, 'AUC Improvements', va='center', rotation='vertical', fontsize=11)
  plt.xlabel('Data sets, sorted by improvements')
  plt.savefig('sample.eps', format='eps',dpi= 1000)
  plt.show()


if __name__ == "__main__":
  run()
