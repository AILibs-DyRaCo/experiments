package dyadranking.experiments.cuda;

import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;
import jaicore.ml.dyadranking.util.AbstractDyadScaler;
import jaicore.ml.dyadranking.util.DyadNormalScaler;

public class PLNetExperiment {
	
	private static final String DATASET_FILE = "./landmarking_mining_combined.txt";

	private static final double TRAIN_RATIO = 0.7d;

	// seed for shuffling the dataset
	private static final long seed = 15;

	private static PLNetDyadRanker ranker;
	private static DyadRankingDataset dataset;

	public static void main(String args[]) {
		
		ranker = new PLNetDyadRanker();

		// load dataset
		dataset = new DyadRankingDataset();
		try {
			dataset.deserialize(new FileInputStream(new File(DATASET_FILE)));
//			DyadRankingDataset datasetAllPairs = getAllPairs(dataset);
//			datasetAllPairs.serialize(new FileOutputStream(new File("landmarking_all_pairs.txt")));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		System.out.println(ranker.getConfiguration());
		AbstractDyadScaler scaler = new DyadNormalScaler();
//	scaler = new DyadStandardScaler();
		Collections.shuffle(dataset, new Random(seed));

		// split data
		DyadRankingDataset trainData = new DyadRankingDataset(dataset.subList(0, (int) (TRAIN_RATIO * dataset.size())));
		DyadRankingDataset testData = new DyadRankingDataset(
				dataset.subList((int) (TRAIN_RATIO * dataset.size()), dataset.size()));
//	
//	trainData = getAllPairs(trainData);
//	trainData = new DyadRankingDataset(trainData.subList(0, N));
//	 DyadRankingDataset trainData = new DyadRankingDataset(dataset.subList(0, N));
//	DyadRankingDataset testData = new DyadRankingDataset(dataset.subList(N, N+M));
		System.out.println("Train data size: " + trainData.size());
//	System.out.println("Test data size: " + testData.size());
		// standardize data
//	scaler.fit(trainData);
//	scaler.transform(trainData);
//	scaler.transform(testData);
		ArrayList<Integer> algorithmIndeces = new ArrayList<Integer>();
		Collections.addAll(algorithmIndeces, 44, 229, 192, 208, 73, 134, 42, 90, 235, 209, 214, 202, 99, 33, 17, 126,
				220, 53, 176, 183, 9, 45, 34, 15, 163, 188, 142, 238, 185, 242, 43, 65, 184, 195, 98, 155, 80, 111, 14,
				88, 0, 171, 207, 97, 251, 248, 247, 182, 57, 52);
//	scaler.transformAlternatives(trainData, algorithmIndeces);
//	scaler.transformAlternatives(testData, algorithmIndeces);
//	
//	System.out.println(testData);

		
		try {

			// train the ranker
			ranker.train(trainData);
			
			try {
				ranker.saveModelToFile("plnetModel");
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			
			double avgKendallTau = 0.0d;
			avgKendallTau = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), testData, ranker);
			System.out.println(
					"Average Kendall's tau for " + ranker.getClass().getSimpleName() + " oos: " + avgKendallTau);
			avgKendallTau = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), trainData, ranker);
			System.out.println(
					"Average Kendall's tau for " + ranker.getClass().getSimpleName() + " is: " + avgKendallTau);
		} catch (TrainingException | PredictionException e) {
			e.printStackTrace();
		}
		
	}
}
