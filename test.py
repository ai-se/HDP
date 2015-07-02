__author__ = 'WeiFu'
from utility import *

def test():
  source_src = "safe1.arff"
  target_src = "tomcat1.arff"
  source_data = loadWekaData(source_src)
  target_data = loadWekaData(target_src)
  # cls = Classifier(classname="weka.classifiers.functions.Logistic")
  cls = autoclass('weka.classifiers.functions.Logistic')()
  cls.buildClassifier(source_data)
  cls.setDebug(True)
  eval = autoclass('weka.classifiers.Evaluation')(source_data)
  eval.evaluateModel(cls, target_data)
  # target_data.num_attributes
  # print(eval.percent_correct)
  print(eval.toSummaryString())
  print(eval.toClassDetailsString())
  # print(eval.class_details())
  # print(eval.area_under_roc(1))
  eval.areaUnderROC(1)

if __name__ == "__main__":
  test()