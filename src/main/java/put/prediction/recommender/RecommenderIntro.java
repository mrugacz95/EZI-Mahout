package put.prediction.recommender;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

class RecommenderIntro {

    private RecommenderIntro() {
    }

    public static void main(String[] args) throws Exception {

        //feed model with data read from file
        DataModel model = new FileDataModel(new File("data/movies.csv"));

        //define user similarity as Pearson correlation coefficient
        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

        //neighborhood = 50 nearest neighbors
        UserNeighborhood neighborhood = new NearestNUserNeighborhood(50, similarity, model);

        // recommender instance - User-based CF
        GenericUserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

        //top 10 recommendation for user 50
        List<RecommendedItem> recommendations = recommender.recommend(50, 10);

        //print recommendation
        for (RecommendedItem recommendation : recommendations) {

            System.out.println(recommendation);
        }

    }
} 
