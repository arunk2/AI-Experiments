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
        

    def load_data(self):
        schema = StructType([
            StructField("label", IntegerType()),
            StructField("text", StringType())
        ])

        df = self.spark.read.csv( "/home/dev/ClassificationMulticlass/dataset.txt", header=False, mode="DROPMALFORMED", schema=schema)
        # df.show()
        return df

    def train_validate(self, df):
        # Split the data into training and test sets (30% held out for testing)
        (training, test) = df.randomSplit([0.7, 0.3])

        # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered")
        hashingTF = HashingTF(numFeatures=10000, inputCol=remover.getOutputCol(), outputCol="features")

        ####################
        # lr = LogisticRegression(maxIter=10, regParam=0.001)
        # pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, lr])
        ####################

        # instantiate the base classifier.
        lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
        # instantiate the One Vs Rest Classifier.
        ovr = OneVsRest(classifier=lr)
        pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, ovr])
        #####################

        # Fit the pipeline to training documents.
        model = pipeline.fit(training)

        # Make predictions on test documents and print columns of interest.
        prediction = model.transform(test)

        # obtain evaluator.
        evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

        # compute the classification error on test data.
        accuracy = evaluator.evaluate(prediction)
        print("Test Error : " + str(1 - accuracy))
        return model

    def train(self):
        df = self.load_data()
        model = self.train_validate(df)
        model.save('Category_Classifier')


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("Category_Classifier")\
        .getOrCreate()

    classifier = ClassifierLR(spark)
    classifier.train()
    
    spark.stop()
