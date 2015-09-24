# __author__ = 'WeiFu'
from __future__ import division, print_function
import time
from hdp import *


def readMatch(src="./result/source_target_match.txt"):
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
             target_name=target_src)
    result.append(temp)
  return result


def getMedian(lst):
  if len(lst) % 2:
    return round(lst[int(len(lst) * 0.5)], 3)
  else:
    return round((lst[int(len(lst) * 0.5 - 0.5)] + lst[int(len(lst) * 0.5 + 0.5)]) / 2, 3)

def getIQR(lst):
  def p(x) : return lst[int(x)]
  n = int(len(lst)*0.25)
  IQR = p(n*3) - p(n*1)
  return IQR


def process(match, target_name, result):
  total = []
  for i in match:
    one_source_result = None
    if i.target_name == target_name:
      one_source_result = [j.result[0] for j in result if j.source_src == i.source_src and j.result != []]
      # put all the results from one source together.
    if not one_source_result:
      continue
    one_median = getMedian(sorted(one_source_result))
    # print(i.source_src, "===>", target_src, one_median)
    total += [one_median]
  if len(total) == 0:
    print("no results for ", target_name)
    return
  total_median = getMedian(sorted(total))
  # print("final ====>", target_name, total_median)
  return total_median


def run1(source_target_match, option):
  out = {}
  original_src = "./dataset"
  datasrc = readsrc(original_src)
  for group, srclst in datasrc.iteritems():
    for target_src in srclst:
      data = loadWekaData(target_src)
      out_wpdp, out_cpdp, out_hdp = [], [], []  # store results for three methods
      target_name = target_src[target_src.rindex("/") + 1:]
      for _ in xrange(10):
        randomized = filter(data, False, "", "weka.filters.unsupervised.instance.Randomize", ["-S", str(_)])
        train = filter(randomized, True, "train", "weka.filters.unsupervised.instance.RemoveFolds",
                       ["-N", "2", "-F", "1", "-S", "1"]) # N : numFolds, F: whichFold to keep, S: is the seed
        test = filter(randomized, True, "test", "weka.filters.unsupervised.instance.RemoveFolds",
                      ["-N", "2", "-F", "2", "-S", "1"])
        # out_wpdp += wpdp(tarin, test)
        # cpdp(group,one)
        temp = hdp(option, target_name, source_target_match)
        if len(temp) == 0:
          continue
        else:
          out_hdp += temp
      dataset = target_src[target_src.rindex("/") + 1:-5]  # get name of datset
      result = process(source_target_match, target_name, out_hdp)
      out[dataset] = out.get(dataset, []) + [result]
      print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"))
  return out


def printout(result_dict):
  out = [result_dict["method"]]
  for key, val in result_dict.iteritems():
    if key == "method":
      continue
    out.append(val)
  printm(out)


def repeat(KSanalyzer, original_src, option, iteration = 20):
  result, temp = {}, {}
  for _ in xrange(iteration):
    if option and (option[option.index("-S") + 1] == "S" or option[option.index("-T") + 1] == "S"):
      small_src = genSmall(option)  # generate small data sets
    if "-EPV" in option:
      source_target_match = KSanalyzer("./EPVSmalldataset","./Smalldataset", option)
    else:
      source_target_match = KSanalyzer(original_src,original_src, option)
    out = run1(source_target_match, option)
    for key, val in out.iteritems():
      temp[key] = temp.get(key, []) + val
  for key, val in temp.iteritems():
    result[key] = [getMedian(sorted(val)),getIQR(sorted(val))]
  return result


def addResult(out, title, new):
  out["method"] = out.get("method") + title
  for key, val in out.iteritems():
    if key == "method":
      continue
    out[key] = out.get(key) + new[key]
  return out


def run(original_src="./dataset", option=["-S", "S", "-T", "S","-EPV",20,"-N", 50]):
  print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"))
  out = {"EQ": ['EQ', 0.583,0.783], "JDT": ['JDT',0.795, 0.767], "LC": ['LC',0.575, 0.655], "ML": ['ML', 0.734,0.692], "PDE": ['PDE',0.684, 0.717],
         "apache": ['apache',0.714, 0.717], "safe": ['safe',0.706, 0.818], "zxing": ['zxing',0.605, 0.650], "ant-1.3": ['ant-1.3',0.609, 0.835],
         "arc": ['arc', 0.670,0.701], "camel-1.0": ['camel-1.0', 0.550,0.639], "poi-1.5": ['poi-1.5',0.707, 0.701],
         "redaktor": ['redaktor',0.744, 0.537], "skarbonka": ['skarbonka',0.569, 0.694], "tomcat": ['tomcat',0.778, 0.818],
         "velocity-1.4": ['velocity-1.4', 0.725,0.391], "xalan-2.4": ['xalan-2.4',0.755, 0.751],
         "xerces-1.2": ['xerces-1.2', 0.624,0.489], "cm1": ['cm1', 0.653,0.717], "mw1": ['mw1', 0.612,0.727], "PC1": ['pc1', 0.787,0.752],
         "PC3": ['pc3', 0.794,0.738], "PC4": ['pc4',0.900, 0.682], "ar1": ['ar1', 0.582,0.734], "ar3": ['ar3', 0.574,0.823],
         "ar4": ['ar4',0.657, 0.816], "ar5": ['ar5',0.804, 0.911], "ar6": ['ar6',0.654, 0.640], "method": ['Target', 'WPDP','HDP-JC']}
  original_src = runPCA()
  # out = addResult(out, ['HDP-Scipy', 'HDP-Scipy-IQR'], repeat(KSanalyzer, original_src, []))
  for num in range(50, 250, 50):
    title = ['N-' + str(num),'N-' + str(num)+'-IQR']
    option[option.index("-N")+1] = num
    out = addResult(out, title, repeat(KSanalyzer, original_src, option))
  printout(out)


def test():
  match = readMatch("./result/Large_Small_match.txt")
  original_src = "./dataset"
  datasrc = readsrc(original_src)
  last = ""
  EPV = {}
  for group,val in datasrc.iteritems():
    for src in val:
      temp = 0
      count = 0
      for i in match:
        if i.target_src == src:
          data = loadWekaData("./Small"+i.target_src[2:])
          num_bug = len(data.attributeToDoubleArray(data.classIndex())) - sum(data.attributeToDoubleArray(data.classIndex()))
          temp += float(num_bug/len(i.attr_target))
          count += 1
      EPV[src[src.rfind("/")+1:src.rfind(".")]] = round(temp/count,3)
  # pdb.set_trace()
  for key, val in EPV.iteritems():
    print(key, ':',val)
  print(EPV)




if __name__ == "__main__":
  random.seed(1)
  # test()
  # readMatch()
  run()
  # printPCA()

