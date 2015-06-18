#__author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random
from os import listdir
from os.path import isfile, join
import weka.core.jvm as jvm
import weka.core.converters
from weka.core.converters import Loader, Saver
from weka.classifiers import Classifier, Evaluation
from weka.experiments import SimpleCrossValidationExperiment
from weka.filters import Filter

class o:
  ID = 0
  def __init__(i, **d):
    o.ID = i.id = o.ID+1
    i.update(**d)

  def update(i, **d): i.__dict__.update(d); return i

  def __getitem__(i, k): return i.__dict__[k]

  def __hash__(i): return i.id

  def __repr__(i):
    keys = [k for k in sorted(i.__dict__.keys()) if k[0] is not "_"]
    show = [":%s %s" % (k, i.__dict__[k]) for k in keys]
    return '{' + ' '.join(show) + '}'


def read(src="./datasetcsv"):
  """
  read data from csv files, return all data in a dictionary

  {'AEEEM':[{name ='./datasetcsv/SOFTLAB/ar6.csv'
             attributes=['ck_oo_numberOfPrivateMethods', 'LDHH_lcom', 'LDHH_fanIn'...]
             instances=[[.....],[.....]]},]
   'MORPH':....
   'NASA':....
   'Relink':....
   'SOFTLAB':....]
  }

  """

  def tofloat(lst):
    for x in lst:
      try:
        yield float(x)
      except ValueError:
        yield x[:-1]

  data = {}
  folders = [i for i in listdir(src) if not isfile(i) and i != ".DS_Store"]
  for f in folders:
    path = join(src, f)
    for val in [join(path, i) for i in listdir(path) if i != ".DS_Store"]:
      d = open(val, "r")
      content = d.readlines()
      attr = content[0].split(",")
      inst = [list(tofloat(row.split(","))) for row in content[1:]]
      data[f] = data.get(f, []) + [o(name=val, attr=attr, data=inst)]
  return data

def readsrc(src="./dataset"):
  """
  read all data files in src folder into dictionary,
  where subfolder src are keys, corresponding file srcs are values
  :param src: the root folder src
  :type src: str
  :return: src of all datasets
  :rtype: dictionary
  """
  data = {}
  subfolder = [ join(src,i) for i in listdir(src) if not isfile(join(src,i))]
  for one in subfolder:
    data[one]= [ join(one,i)for i in listdir(one) if isfile(join(one,i)) and i != ".DS_Store"]
  # print(data)
  return data

def readarff(src = "./dataset/AEEEM/EQ.arff"):
  """
  read each arff and return header and content
  :param src: src of arff file
  :type src : str
  :return: header and content of each arff file
  :rtype:tuple (str, list)
  """
  f = open(src, "r")
  content,arffheader  = [],[]
  while True:
    line = f.readline()
    if not line:
      break
    elif "@" in line:
      arffheader += [line]
      if "data" in line:
        continue
    else:
      if len(line) <10:
        continue
      if line[-1] !="\n":
        line+="\n"
      content +=[line]
  return "".join(arffheader), content

def writearff(data,name,src = "./exp"):
  """
  """
  wf = open(src+"/"+name+".arff","w")
  wf.write(data)

def wekaExp( datasets=["./dataset/SOFTLAB/ar3.arff"], run=500, fold=2):
  if not jvm.started: jvm.start()
  classifiers = [Classifier(classname="weka.classifiers.functions.Logistic")]
  result = "exp.arff"
  exp = SimpleCrossValidationExperiment(
        classification=True,
        runs = run,
        folds = fold,
        datasets = datasets,
        classifiers = classifiers,
        result= result
  )
  exp.setup()
  exp.run()
  loader = weka.core.converters.loader_for_file(result)
  data = loader.load_file(result)
  from weka.experiments import Tester, ResultMatrix
  matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
  tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
  tester.resultmatrix = matrix
  # comparison_col = data.attribute_by_name("Percent_correct").index
  comparison_col = data.attribute_by_name("Area_under_ROC").index
  tester.instances = data
  pdb.set_trace()
  print(tester.header(comparison_col))
  print(tester.multi_resultset_full(0, comparison_col))

def wekaCALL(train, test, train_attr = [], test_attr = [], isHDP = False):
  """
  weka wrapper to train and test based on the datasets
  :param train: traininng data
  :type train: str(src)
  :param test: testing data
  :type test: str(src)
  """
  def getIndex(data,used_attr):
    del_attr = []
    for k,attr in enumerate(data.attributes()):
      temp = str(attr).split(" ")
      if temp[1] not in used_attr:
        del_attr +=[k]
    return del_attr

  def delAttr(data,index):
    order = sorted(index, reverse=True)
    for i in order[1:]: # delete from big index, except for the class attribute
      data.delete_attribute(i)
    return data

  if not jvm.started: jvm.start()
  loader = Loader(classname="weka.core.converters.ArffLoader")
  train_data = loader.load_file(train)
  test_data = loader.load_file(test)
  train_data.class_is_last()
  test_data.class_is_last()
  cls = Classifier(classname="weka.classifiers.functions.Logistic")
  if isHDP:
    train_del_attr = getIndex(train_data, train_attr)
    test_del_attr = getIndex(test_data,test_attr)
    train_data = delAttr(train_data,train_del_attr)
    test_data = delAttr(test_data,test_del_attr)
  cls.build_classifier(train_data)
  eval = Evaluation(train_data)
  eval.test_model(cls,test_data)
  test_data.num_attributes
  # print(eval.percent_correct)
  # print(eval.summary())
  # print(eval.class_details())
  # print(eval.area_under_roc(1))
  return eval.area_under_roc(1)



def filter(data_src = "./dataset/AEEEM/EQ.arff", option = ["-R", "first-3,last"]):
  if not jvm.started: jvm.start()
  loader = Loader(classname="weka.core.converters.ArffLoader")
  data = loader.load_file(data_src)
  remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options = option)
  remove.inputformat(data)
  filtered = remove.filter(data)
  saver = Saver(classname="weka.core.converters.ArffSaver")
  saver.save_file(filtered,"./dataset/AEEEM/EQ_FFF.arff")
  print(filtered)
  return filtered

def attributeSelection():
  if not jvm.started: jvm.start()
  loader = Loader(classname="weka.core.converters.ArffLoader")
  data = loader.load_file("./dataset/AEEEM/EQ.arff")
  data.class_is_last()

  from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
  search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-D", "1", "-N", "9"])
  evaluator = ASEvaluation(classname="weka.attributeSelection.ChiSquaredAttributeEval")
  attsel = AttributeSelection()
  attsel.search(search)
  attsel.evaluator(evaluator)
  attsel.select_attributes(data)

  print("# attributes: " + str(attsel.number_attributes_selected))
  print("attributes: " + str(attsel.selected_attributes))
  pdb.set_trace()
  print("result string:\n" + attsel.results_string)

if __name__ == "__main__":
  attributeSelection()