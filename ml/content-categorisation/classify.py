from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row, SparkSession
from pyspark.ml.feature import StopWordsRemover

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

from pyspark.ml.classification import LogisticRegression, OneVsRest, OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import datetime


class ClassifierLR(object):

    def __init__(self, spark):
        super(ClassifierLR, self).__init__()
        self.spark = spark
        self.newmodel = PipelineModel.load("Category_Classifier")

    def infer(self, validateSet):
        # Prepare test documents, which are unlabeled.
        test = self.spark.createDataFrame(validateSet, ["id", "text"])
        prediction = self.newmodel.transform(test)
        selected = prediction.select("text", "prediction")

        label = []
        for row in selected.collect():
            label.append(str(row[1]))
            
        return label

    def infer_rdd(self, rdd):
        # Prepare test documents, which are unlabeled.
        test = self.spark.createDataFrame(rdd, ["text"])
        prediction = self.newmodel.transform(test)
        selected = prediction.select("text", "prediction")
        return selected.rdd.map(tuple)


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Category_Classifier")\
        .getOrCreate()

    validateSet = [
        (1,'suzuki grand vitara lose road capabilities suzuki'),
        (2,'bmw patent images'),
        (3,'indian rolls royce owners'),
        (4,'firearms illegal drugs exmarawi mayor fajad umpar salics house'),
        (5,'thieves steal lakh worth cloths theft cases karkala udayavani'),
        (6,'shootout mandaue drug lord bakilid aquino shooting sunstar')
    ]

    classifier = ClassifierLR(spark)
    classifier.infer(validateSet)
    
    spark.stop()
