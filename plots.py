from __future__ import division
# __author__ = 'WeiFu'
from matplotlib import pyplot as plt
import numpy as np
import sys, pdb


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



def run(N = 4):
  data = getData()
  samplesize= [50,100,150,200]
  x = np.linspace(0,27,28)
  for plotNum in range(N):
    onePlot ={}
    for key, val in data.iteritems():
      onePlot[key] =[val[0],val[plotNum+1],val[plotNum+1]-val[0]]
    onePlot_sorted = sorted(onePlot.items(), key = lambda x:x[-1][-1])
    base = [ i[-1][0] for i in onePlot_sorted]
    proposed = [i[-1][1] for i in onePlot_sorted]
    print plotNum
    # plt.subplot(4,1,plotNum+1)
    plt.plot(x,base,'ko-',label = 'All')
    plt.plot(x,proposed,'ro-',label = 'N= '+str(samplesize[plotNum]))
    plt.legend(fontsize = 'xx-small')
    plt.xlim([0,28])
  plt.xlabel('Data sets, sorted by improvements')
  plt.ylabel('Improvements')
  plt.show()




if __name__ == "__main__":
  run()
