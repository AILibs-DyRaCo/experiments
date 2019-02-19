package dyadranking.zeroshotml.experiments;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;
import jaicore.ml.dyadranking.util.DyadNormalScaler;

public class PLNetEvaluation {
	
	private static final String TEST_DATA = "datasets/zeroshot/SMORBFtest.dr";
	
	private static final String PLNET = "datasets/zeroshot/SMORBFPLNet.plnet.zip";
	
	public static void main(String[] args) throws IOException, PredictionException {
		
		File dataFile = new File(TEST_DATA);
		
		DyadRankingDataset data = new DyadRankingDataset();
		data.deserialize(new FileInputStream(dataFile));
		
		DyadNormalScaler scaler = new DyadNormalScaler();
		scaler.fit(data);
		scaler.transformAlternatives(data);
		
		PLNetDyadRanker plNet = new PLNetDyadRanker();
		plNet.loadModelFromFile(PLNET);
		
		double avgKendallTau = 0.0d;
		avgKendallTau = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), data, plNet);
		
		System.out.println("Kendall's tau: " + avgKendallTau);
	}

}
