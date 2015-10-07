from __future__ import print_function, division
# __author__ = 'WeiFu'
import sys, pdb, random, math
from utility import *
from experiment import  *
import matplotlib.pyplot as plt
import numpy as np

def space(p, columns, N, num_of_bins):
  def ps(myspace):
    prob = []
    for col in myspace:
      for bin in col:
        temp_p = 1 - sum([i[1] for i in bin]) / len(bin)  # prob of defective instances
        prob.append(temp_p)
    all = sum(prob)
    f2 = [x / all for x in prob]
    return [p * 1 / num_of_bins * x for x in f2]

  last, dist, out = 0, int(math.ceil(len(columns[0]) / N)), []
  cut = [(j + 1) * dist for j in range(N) if (j + 1) * dist < len(columns[0])]
  cut.extend([len(columns[0])])
  space = []
  for col in columns:
    bins = []
    last = 0
    for c in cut:
      bins.append(col[last:c])
      last = c
    space.append(bins)
  return ps(space)


def test(w):
  n = found = 0
  points = {}
  while found < 0.90:
    n += 10
    found = 1 - (1 - w) ** n
    points[str(n)] = points.get(str(n), []) + [round(found,3)]
  return points


def chops(match , source_src, selected_attr=[], N=3):
  arff = loadWekaData(source_src)
  attributes = [str(i)[str(i).find("@attribute") + len("@attribute") + 1:str(i).find("numeric") - 1] for i in
                enumerateToList(arff.enumerateAttributes())]  # exclude the label
  attributes_index = [i for i, attr in enumerate(attributes) if attr in selected_attr]
  columns = [sorted(zip(arff.attributeToDoubleArray(i), arff.attributeToDoubleArray(arff.classIndex())), reverse=True)
             for i in attributes_index]  # exclude the class label
  num_of_bins = N ** len(attributes_index)
  p = numBuggyInstance(arff) / arff.size()
  w = sum(space(p, columns, N, num_of_bins))
  distribution = test(w)
  return distribution

def run():
  source_target_match = readMatch("./result/Sim3_source_target_match0727.txt")
  original_src = "./dataset"
  datasrc = readsrc(original_src)
  for group, srclst in datasrc.iteritems():
    for target_src in srclst:
      target_name = target_src[target_src.rindex("/") + 1:]
      print("target:", target_name,"*"*10)
      for i in source_target_match:
        if i.target_name == target_name:
          result ={"method":["N"]}
          result['method'] = result.get('method') +[i.source_src[i.source_src.rindex("/")+1:]]
          for key,val in chops([],i.source_src,i.attr_source).iteritems():
            result[key] = result.get(key,[key])+val
          printout(result)
          pdb.set_trace()


def plot(result):
  # color = ['r-','k-','b-','b^','g-','y-','c-','m-']
  # labels = ['WHICH','Tuned_WHICH','manualUp','manualDown','minimum','best','Tuned_CART','CART']
  color = ['r-', 'k-', 'b-', 'g-', 'y-', 'c-', 'm-']
  labels = ['Prob', 'manualUp', 'manualDown', 'minimum', 'best', 'CART', 'C4.5']
  plt.figure(1)
  x = result
  plt.plot(x[0], x[1], color[0], label=labels[0])
  plt.xlabel("N(sample size)")
  plt.ylabel("Prob")
  # plt.title("Effort-vs-PD")
  plt.ylim(0, 1)
  plt.legend(loc='best')
  plt.show()

def distribution():
  source_target_match = readMatch("./result/Large_Small_match.txt")
  original_src = "./dataset"
  datasrc = readsrc(original_src)
  result = {}
  for i in source_target_match:
    result[len(i.attr_source)]=result.get(len(i.attr_source),0) +1
  for key,val in result.iteritems():
    print("dim="+str(key),":",str(val))



def test():
  w = 0.003
  out = []
  index = []
  for i in xrange(0,1000,10):
    p = 1-(1-w)**i
    out.append(p)
    index.append(i)
  pdb.set_trace()
  plot([np.array(index),np.array(out)])
  print(out)
if __name__ == "__main__":
  # run()
  # test()
  distribution()

