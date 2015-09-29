from __future__ import print_function, division
# __author__ = 'WeiFu'
import sys, pdb, random,math
from utility import *

def space(p,columns,N,num_of_bins):
  def ps(myspace):
    prob = []
    for col in myspace:
      for bin in col:
        temp_p = 1- sum([i[1] for i in bin])/len(bin) # prob of defective instances
        prob.append(temp_p)
    all = sum(prob)
    f2 = [x/all for x in prob]
    return [p*1/num_of_bins*x for x in f2]

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



def chops(source_src, selected_attr, N = 3):
  arff = loadWekaData(source_src)
  attributes = [str(i)[str(i).find("@attribute") + len("@attribute") + 1:str(i).find("numeric") - 1] for i in
                    enumerateToList(arff.enumerateAttributes())]  # exclude the label
  attributes_index = [i for i, attr in enumerate(attributes) if attr in selected_attr]
  columns = [sorted(zip(arff.attributeToDoubleArray(i), arff.attributeToDoubleArray(arff.classIndex())), reverse = True)
                    for i in attributes_index]  # exclude the class label
  num_of_bins = N * len(attributes_index)
  pdb.set_trace()
  p = numBuggyInstance(arff)/arff.size()
  w = sum(space(p, columns,N, num_of_bins))
  print(w)
  pdb.set_trace()
  print("Done")




  pdb.set_trace()
  pass
