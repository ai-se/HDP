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
             target_src=target_src)
    result.append(temp)
  return result


def getMedian(lst):
  if len(lst) % 2:
    return round(lst[int(len(lst) * 0.5)],3)
  else:
    return round((lst[int(len(lst) * 0.5 - 0.5)] + lst[int(len(lst) * 0.5 + 0.5)]) / 2,3)


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
    # print(i.source_src, "===>", target_src, one_median)
    total += [one_median]
  if len(total) == 0:
    print("no results for ", target_src)
    return
  total_median = getMedian(sorted(total))
  print("final ====>", target_src, total_median)
  return total_median


def run1(source_target_match, datasrc, use_small_source):
  out = {}
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
        temp = hdp(use_small_source, target_src, source_target_match)
        if len(temp) == 0:
          continue
        else:
          out_hdp += temp
      dataset = target_src[target_src.rindex("/")+1:-5]
      result = process(source_target_match, target_src, out_hdp)
      out[dataset] = out.get(dataset,[])+ [result]
      print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"))
  # printout(out)
  return out

def printout(result_dict):
  out = [result_dict["method"]]
  for key,val in result_dict.iteritems():
    if key== "method":
      continue
    out.append(val)
  printm(out)

def repeat(KSanalyzer,original_src,option, datasrc, use_small_source):
  result,temp={},{}
  for _ in xrange(20):
    if use_small_source:
      small_src = runSmall(option)
    source_target_match = KSanalyzer(original_src,option)
    out = run1(source_target_match, datasrc, use_small_source)
    for key, val in out.iteritems():
      temp[key] = temp.get(key,[])+ val
  for key, val in temp.iteritems():
    result[key]=[getMedian(sorted(val))]
  return result

def addResult(out,method,new):
  out["method"] = out.get("method")+[method]
  for key, val in out.iteritems():
    if key == "method":
      continue
    out[key] = out.get(key)+new[key]
  return out


def run(original_src="./dataset",option=["-S", "S", "-T", "S", "-N", 200]):
  print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"))
  out = {"EQ":['EQ',0.783],"JDT":['JDT',0.767],"LC":['LC',0.655],"ML":['ML',0.692],"PDE":['PDE', 0.717],
        "apache":['apache',0.717],"safe":['safe',0.818],"zxing":['zxing',0.650],"ant-1.3":['ant-1.3', 0.835],
        "arc":['arc', 0.701],"camel-1.0":['camel-1.0', 0.639],"poi-1.5":['poi-1.5',0.701],"redaktor":['redaktor',0.537],
        "skarbonka":['skarbonka', 0.694],"tomcat":['tomcat', 0.818],"velocity-1.4":['velocity-1.4',0.391],
        "xalan-2.4":['xalan-2.4',0.751], "xerces-1.2":['xerces-1.2',0.489],"cm1":['cm1', 0.717],"mw1":['mw1',0.727],
        "PC1":['pc1', 0.752],"PC3":['pc3',0.738], "PC4":['pc4', 0.682],"ar1":['ar1', 0.734],"ar3":['ar3', 0.823],
        "ar4":['ar4', 0.816],"ar5":['ar5', 0.911],"ar6":['ar6',0.640],"method":['Target','HDP-JC']}
  # src = runPCA()
  datasrc = readsrc(original_src)
  source_target_match = KSanalyzer(original_src, [])  # run JC's experiment
  out = addResult(out,'HDP-Scipy',repeat(KSanalyzer, original_src,[],datasrc, False))
  out = addResult(out,'N-200',repeat(KSanalyzer, original_src,option,datasrc, True))
  printout(out)
  print(time.strftime("%a, %d %b %Y %H:%M:%S +0000"))


  # # repeat(source_target_match, datasrc, False)
  # source_target_match = KSanalyzer(original_src, option) # reduced_instance
  # repeat(source_target_match, datasrc, True)


if __name__ == "__main__":
  # readMatch()
  run()
