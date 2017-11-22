from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import Row, SparkSession
from pyspark.ml.feature import StopWordsRemover

from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

from pyspark.ml.classification import LogisticRegression, OneVsRest, OneVsRestModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import datetime


def quiet_logs( sc ):
    '''
    Mute spark logs
    '''
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )


class ClassifierLR(object):

    def __init__(self, spark):
        super(ClassifierLR, self).__init__()
        self.spark = spark
        self.newmodel = PipelineModel.load("Category_Classifier")


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

    quiet_logs(spark)

    validateSet = [
        (1,'suzuki grand vitara lose road capabilities suzuki'),
        (2,'bmw patent images'),
        (3,'indian rolls royce owners'),
        (4,'firearms illegal drugs exmarawi mayor fajad umpar salics house'),
        (5,'thieves steal lakh worth cloths theft cases karkala udayavani'),
        (6,'shootout mandaue drug lord bakilid aquino shooting sunstar')
    ]

    classifier = ClassifierLR(spark)
    
    # classifier.train()
    print "Start : %s" % datetime.datetime.today()
    classifier.infer(validateSet)
    print "End   : %s" % datetime.datetime.today()
    
    spark.stop()
