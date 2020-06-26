package main.scala.djgarcia

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature._

import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.RandomNoise
import java.io.{File, PrintWriter}

object runEnsembles {

  def main(arg: Array[String]) {

    //Basic setup
    val jobName = "MLlib Ensembles"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)

    //Log level
    sc.setLogLevel("ERROR")    

    //Load train and test
    //Para la máquina virtual
    val pathTrain = "file:///home/spark/datasets/higgsMaster-Train.data"

        //para el cluster
    //val pathTrain = "/user/datasets/master/higgs/higgsMaster-Train.data"

    val rawDataTrain = sc.textFile(pathTrain)
   
    //Para la máquina virtual
    val pathTest = "file:///home/spark/datasets/higgsMaster-Test.data"

    //para el cluster
    //val pathTest = "/user/datasets/master/higgs/higgsMaster-Test.data"

    val rawDataTest = sc.textFile(pathTest)
    

    val train = rawDataTrain.map { line =>
      val array = line.split(",")
      val arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.persist

    train.count
    train.first

    val test = rawDataTest.map { line =>
      val array = line.split(",")
      val arrayDouble = array.map(f => f.toDouble)
      val featureVector = Vectors.dense(arrayDouble.init)
      val label = arrayDouble.last
      LabeledPoint(label, featureVector)
    }.persist

    test.count
    test.first

    //Load train and test with KeelParser

    /*val converter = new KeelParser(sc, "file:///home/spark/datasets/susy.header")
    val train = sc.textFile("file:///home/spark/datasets/susy-10k-tra.data", 10).map(line => converter.parserToLabeledPoint(line)).persist
    val test = sc.textFile("file:///home/spark/datasets/susy-10k-tst.data", 10).map(line => converter.parserToLabeledPoint(line)).persist
*/

    

	//Ejemplos de uso de los 2 métodos

	// ROS 100% --> igualamos la proporción de las clases

	val trainROS = ROS(train, 1.0)

	val classInfoROS = trainROS.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()


	// RUS

	//val trainRUS = RUS(train)

	//val classInfoRUS = trainRUS.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()
/*
	 //HME-BD Noise Filter

	    val nTrees = 130
	    val maxDepthRF = 20
	    val partitions = 6

	    val hme_bd_model_noisy = new HME_BD(trainROS, nTrees, partitions, maxDepthRF, 48151623)

	    val hme_bd_noisy = hme_bd_model_noisy.runFilter()

	    hme_bd_noisy.persist()

	    hme_bd_noisy.count()


	//// Noise filter

	    val k = 11 //number of neighbors

	    val ncnedit_bd_model = new NCNEdit_BD(trainRUS, k)

	    val ncnedit_bd = ncnedit_bd_model.runFilter()



 //RNG Noise Filter

    val order = true // Order of the graph (true = first, false = second)
    val selType = true // Selection type (true = edition, false = condensation)

    val rng_bd_model = new RNG_BD(trainRUS, order, selType)

    val rng_bd = rng_bd_model.runFilter()

    rng_bd.persist()

    rng_bd.count()


*/

    //-----Instance Selection-----//


    //FCNN

    val k = 7 //number of neighbors
    val fcnn_mr_model = new FCNN_MR(trainROS, k)

    val fcnn_mr = fcnn_mr_model.runPR()

    fcnn_mr.persist()

    fcnn_mr.count()
/*


    //Decision tree

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    var impurity = "gini"
    var maxDepth = 5
    var maxBins = 32

    val modelDT = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPredsDT = test.map { point =>
      val prediction = modelDT.predict(point.features)
      (point.label, prediction)
    }
    val testAccDT = 1 - labelAndPredsDT.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy DT= $testAccDT")
*/

    //Random Forest

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    var numClasses = 2
    var categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    var impurity = "gini"
    var maxDepth = 25
    var maxBins = 32

    val modelRF = RandomForest.trainClassifier(fcnn_mr, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPredsRF = test.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }
    val testAccRF = 1 - labelAndPredsRF.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Accuracy RF= $testAccRF")

/*
    //PCARD

    import org.apache.spark.mllib.tree.PCARD

    val cuts = 5
    val trees = 10

    val pcardTrain = PCARD.train(ncnedit_bd, trees, cuts)

    val pcard = pcardTrain.predict(test)

    val labels = test.map(_.label).collect()

    var cont = 0

    for (i <- labels.indices) {
      if (labels(i) == pcard(i)) {
        cont += 1
      }
    }

    val testAcc = cont / labels.length.toFloat

    println(s"Test Accuracy = $testAcc")


    val predsAndLabels = sc.parallelize(pcard).zipWithIndex.map { case (v, k) => (k, v) }.join(test.zipWithIndex.map { case (v, k) => (k, v.label) }).map(_._2)

*/
    //Metrics

    import org.apache.spark.mllib.evaluation.MulticlassMetrics

    val metrics = new MulticlassMetrics(labelAndPredsRF)
    val cm = metrics.confusionMatrix
    val TPR=(cm(0,0)/(cm(0,0)+cm(0,1)))
    val TNR=(cm(1,1)/(cm(1,1)+cm(1,0)))
    println("TPR = " + TPR)
    println("TNR = " + TNR)
   



    //Write Results
    //para maquina virtual
    val writer = new PrintWriter("/home/spark/results7.txt")

    //para cluster
    //val writer = new PrintWriter("/home/x77147677/results6.txt")
    writer.write(
      "Accuracy: " + testAccRF + "\n" +
      "TPR * TNR:" + TPR*TNR + "\n" +
      "TPR: " + TPR + "\n" +
      "TNR: " + TNR
    )
    writer.close()

  }
	// *** DEFINICIÓN DE LAS FUNCIONES
	//Definición de las funciones --> copiar y pegar en el script

	// ROS

	def ROS(train: RDD[LabeledPoint], overRate: Double): RDD[LabeledPoint] = {
	  var oversample: RDD[LabeledPoint] = train.sparkContext.emptyRDD

	  val train_positive = train.filter(_.label == 1)
	  val train_negative = train.filter(_.label == 0)
	  val num_neg = train_negative.count().toDouble
	  val num_pos = train_positive.count().toDouble

	  if (num_pos > num_neg) {
	    val fraction = (num_pos * overRate) / num_neg
	    oversample = train_positive.union(train_negative.sample(withReplacement = true, fraction))
	  } else {
	    val fraction = (num_neg * overRate) / num_pos
	    oversample = train_negative.union(train_positive.sample(withReplacement = true, fraction))
	  }
	  oversample.repartition(train.getNumPartitions)
	}


	// RUS

	def RUS(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
	  var undersample: RDD[LabeledPoint] = train.sparkContext.emptyRDD

	  val train_positive = train.filter(_.label == 1)
	  val train_negative = train.filter(_.label == 0)
	  val num_neg = train_negative.count().toDouble
	  val num_pos = train_positive.count().toDouble

	  if (num_pos > num_neg) {
	    val fraction = num_neg / num_pos
	    undersample = train_negative.union(train_positive.sample(withReplacement = false, fraction))
	  } else {
	    val fraction = num_pos / num_neg
	    undersample = train_positive.union(train_negative.sample(withReplacement = false, fraction))
	  }
	  undersample
	}

	// *** FIN DEFINICIÓN DE LAS FUNCIONES
}