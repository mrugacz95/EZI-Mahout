import org.apache.mahout.cf.taste.impl.model.file.FileDataModel
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity
import java.io.File
fun main() {
    val dataModel = FileDataModel(File("data/movies.csv"))

    val similarity = EuclideanDistanceSimilarity(dataModel)

    val neighborhood = ThresholdUserNeighborhood(0.7, similarity, dataModel)

    val recommender = GenericUserBasedRecommender(dataModel, neighborhood, similarity)

    for(userID in dataModel.userIDs) {
        println("$userID:")
        val recommendations = recommender.recommend(userID, 3)
        for(recommendation in recommendations) {
            println("${recommendation.itemID} ${recommendation.value}")
        }
    }

//    943:
//    258 5.0
//    300 4.3333335
//    751 4.0

}