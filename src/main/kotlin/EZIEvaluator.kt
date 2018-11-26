import org.apache.mahout.cf.taste.eval.RecommenderBuilder
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood
import org.apache.mahout.cf.taste.similarity.UserSimilarity
import org.apache.mahout.common.RandomUtils
import java.io.File

fun main(args: Array<String>) {

    RandomUtils.useTestSeed()
    val model = FileDataModel(File("data/movies.csv"))


    val similarities = arrayListOf<UserSimilarity>(
        EuclideanDistanceSimilarity(model),
        PearsonCorrelationSimilarity(model),
        TanimotoCoefficientSimilarity(model)
    )

    for (similarity in similarities) {

        val neighbourhoods = arrayListOf<UserNeighborhood>(
            ThresholdUserNeighborhood(0.5, similarity, model),
            ThresholdUserNeighborhood(0.7, similarity, model),
            NearestNUserNeighborhood(5, similarity, model),
            NearestNUserNeighborhood(9, similarity, model)
        )
        for (neighbourhood in neighbourhoods) {


            val evaluator = RMSRecommenderEvaluator()

            val recommenderBuilder = RecommenderBuilder { dataModel ->
                GenericUserBasedRecommender(dataModel, neighbourhood, similarity)
            }
            val stats = evaluator.evaluate(
                recommenderBuilder,
                null,
                model,
                0.7,
                1.0
            )
            println("$similarity $neighbourhood ${stats}")
        }
    }
}

// EuclideanDistance ThresholdUserNeighborhood 0.6583197025604112 0.5
// EuclideanDistance ThresholdUserNeighborhood 0.08213457269587031 0.7
// EuclideanDistance NearestNUserNeighborhood 0.335943037832251 5
// EuclideanDistance NearestNUserNeighborhood 0.3717223053835595 9
// PearsonCorrelation ThresholdUserNeighborhood 0.7856241735123759 0.5
// PearsonCorrelation ThresholdUserNeighborhood 0.7235490966950776 0.7
// PearsonCorrelation NearestNUserNeighborhood 0.7964843683659166 5
// PearsonCorrelation NearestNUserNeighborhood 0.775098066135007 9
// TanimotoCoefficient ThresholdUserNeighborhood 0.8826442251931073 0.5
// TanimotoCoefficient ThresholdUserNeighborhood NaN 0.7
// TanimotoCoefficient NearestNUserNeighborhood 1.0991845262191104 5
// TanimotoCoefficient NearestNUserNeighborhood 1.075459373503826 9

// Best: EuclideanDistance Threshold 0.7 score: 0.08213457269587031
