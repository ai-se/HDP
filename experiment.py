# __author__ = 'WeiFu'
from __future__ import division, print_function
from utility import *
from wpdp import *
from cpdp import *
from hdp import *
import time


def readMatch(src="./result/PCA_source_target_match0710.txt"):
  def getStrip(lst):
    result = []
    for one in lst:
      result.append(one[one.index("'") + 1:one.rindex("'")])
    return result

  result = []
  f = open(src, "r")
  X = f.readlines()[0].split("}, {")
  for each in X:
    attr_source = getStrip(
      each[each.index("attr_source") + len("attr_source") + 2:each.index("attr_target") - 3].split(","))
    attr_target = getStrip(each[each.index("attr_target") + len("attr_target") + 2:each.index("group") - 3].split(","))
    group = each[each.index("group") + len("group") + 1:each.index("id") - 2]
    score = float(each[each.index("score") + len("score") + 2:each.index("source_src") - 2])
    source_src = (each[each.index("source_src") + len("source_src") + 1:each.index("target_src") - 2])
    target_src = (each[each.index("target_src") + len("target_src") + 1:])
    temp = o(score=score, attr_source=attr_source, attr_target=attr_target, source_src=source_src,
             target_src=target_src)
    result.append(temp)
  return result
  # pdb.set_trace()


def getMedian(lst):
  if len(lst) % 2:
    return round(lst[int(len(lst) * 0.5)], 3)
  else:
    return round((lst[int(len(lst) * 0.5 - 0.5)] + lst[int(len(lst) * 0.5 + 0.5)]) / 2, 3)


def process(match, target_src, result):
  total = []
  for i in match:
    one_source_result = None
    if i.target_src == target_src:
      one_source_result = [j.result[0] for j in result if
                           j.source_src == i.source_src and j.result != []]  # put all the results from one source
      # together.
    if not one_source_result:
      continue
    one_median = getMedian(sorted(one_source_result))
    print(i.source_src, "===>", target_src, one_median)
    total += [one_median]
  if len(total) == 0:
    print("no results for ", target_src)
    return
  total_median = getMedian(sorted(total))
  print("final ====>", target_src, total_median)
  return total_median


def run(num_of_component, src="./dataset"):
  print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"))
  out = {}
  src = runPCA(num_of_component)
  datasrc = readsrc(src)
  source_target_match = KSanalyzer(src, [])
  # source_target_match = KSanalyzer(src, ["-S","L","-T","L","-N",200]) # to do online test ,you need to uncomment
  # pdb.set_trace()
  # source_target_match = readMatch()
  for group, srclst in datasrc.iteritems():
    for target_src in srclst:
      random.seed(1)
      data = loadWekaData(target_src)
      out_wpdp, out_cpdp, out_hdp = [], [], []  # store results for three methods
      for _ in xrange(10):
        randomized = filter(data, False, "", "weka.filters.unsupervised.instance.Randomize", ["-S", str(_)])
        train = filter(randomized, True, "train", "weka.filters.unsupervised.instance.RemoveFolds",
                       ["-N", "2", "-F", "1", "-S", "1"])
        test = filter(randomized, True, "test", "weka.filters.unsupervised.instance.RemoveFolds",
                      ["-N", "2", "-F", "2", "-S", "1"])
        # out_wpdp += wpdp(tarin, test)
        # cpdp(group,one)
        temp = hdp(target_src, source_target_match)
        if len(temp) == 0:
          continue
        else:
          out_hdp += temp
      dataset = target_src[target_src.rindex("/") + 1:-5]
      result = process(source_target_match, target_src, out_hdp)
      out[dataset] = out.get(dataset, []) + [result]
      print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"))
  return out


def addResult(out, method, new):
  out["method"] = out.get("method") + [method]
  for key, val in out.iteritems():
    if key == "method":
      continue
    out[key] = out.get(key) + new[key]
  return out


def printout(result_dict):
  out = [result_dict["method"]]
  for key, val in result_dict.iteritems():
    if key == "method":
      continue
    out.append(val)
  printm(out)


def printPCA():
  out = {"EQ": ['EQ', 0.783, 0.432, 0.835], "JDT": ['JDT', 0.767, 0.472, 0.614],
         "LC": ['LC', 0.655, 0.509, 0.774], "ML": ['ML', 0.692, 0.518, 0.8],
         "PDE": ['PDE', 0.717, 0.515, 0.74], "apache": ['apache', 0.717, 0.47, 0.746],
         "safe": ['safe', 0.818, 0.528, 0.772], "zxing": ['zxing', 0.650, 0.557, 0.631],
         "ant-1.3": ['ant-1.3', 0.835, 0.523, 0], "arc": ['arc', 0.701, 0.476, 0],
         "camel-1.0": ['camel-1.0', 0.639, 0.484, 0], "poi-1.5": ['poi-1.5', 0.701, 0.499, 0],
         "redaktor": ['redaktor', 0.537, 0.536, 0.361], "skarbonka": ['skarbonka', 0.694, 0.52, 0.736],
         "tomcat": ['tomcat', 0.818, 0.457, 0], "velocity-1.4": ['velocity-1.4', 0.391, 0.527, 0],
         "xalan-2.4": ['xalan-2.4', 0.751, 0.452, 0], "xerces-1.2": ['xerces-1.2', 0.489, 0.494, 0],
         "cm1": ['cm1', 0.717, 0.513, 0.702], "mw1": ['mw1', 0.727, 0.602, 0.482],
         "PC1": ['pc1', 0.752, 0.493, 0.263], "PC3": ['pc3', 0.738, 0.508, 0.76],
         "PC4": ['pc4', 0.682, 0.546, 0.585], "ar1": ['ar1', 0.734, 0.504, 0.738],
         "ar3": ['ar3', 0.823, 0.481, 0.486], "ar4": ['ar4', 0.816, 0.454, 0.549],
         "ar5": ['ar5', 0.911, 0.554, 0.423], "ar6": ['ar6', 0.640, 0.578, 0.295],
         "method": ['Target', 'HDP-JC', 'PCA-ALL*0.15-apache', 'PCA-2-apache']}
  for num in [2,4,8,-1]:
    title = 'PCA-' + str(num)  + '-Scipy'
    out = addResult(out, title, run(num))

  printout(out)


if __name__ == "__main__":
  # readMatch()
  # run()
  printPCA()