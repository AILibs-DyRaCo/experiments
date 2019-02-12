package dyadranking.zeroshotml.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.regex.Pattern;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.util.DyadNormalScaler;
import jaicore.ml.dyadranking.zeroshot.inputoptimization.NegIdentityInpOptLoss;
import jaicore.ml.dyadranking.zeroshot.inputoptimization.PLNetInputOptimizer;
import jaicore.ml.dyadranking.zeroshot.util.InputOptListener;
import jaicore.ml.dyadranking.zeroshot.util.ZeroShotUtil;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class ZeroShotCurvesExperiment {
	
	private static final String DATA_PATH = "datasets/zeroshot/J48train.dr";
	
	private static final String PLNET_PATH = "datasets/zeroshot/J48PLNet.plnet.zip";
	
	private static final String EVAL_DATA_PATH = "datasets/test";
	
	private static final String DATASET_METAFEAT_TABLE = "`dataset_metafeatures_mirror`";
	
	private static final int[] DATASETS_TEST = { 5, 6, 36, 38, 44, 46 };
	
	private static final int NUM_HYPERPARS = 2;

	private static final double LEARNING_RATE = 0.0005;

	private static final int NUM_ITERATIONS = 200;
	
	private static final int SEED = 42;
	
	private static Map<String, String> datasetIdMap = new HashMap<String, String>() {{
		put("12", "dataset_12_mfeat-factors.arff");
		put("14", "dataset_14_mfeat-fourier.arff");
		put("16", "dataset_16_mfeat-karhunen.arff");
		put("18", "dataset_18_mfeat-morphological.arff");
		put("20", "dataset_20_mfeat-pixel.arff");
		put("21", "dataset_21_car.arff");
		put("22", "dataset_22_mfeat-zernike.arff");
		put("23", "dataset_23_cmc.arff");
		put("24", "dataset_24_mushroom.arff");
		put("26", "dataset_26_nursery.arff");
		put("28", "dataset_28_optdigits.arff");
		put("3", "dataset_3_kr-vs-kp.arff");
		put("30", "dataset_30_page-blocks.arff");
		put("32", "dataset_32_pendigits.arff");
		put("5", "dataset_5_arrhythmia.arff");
		put("6", "dataset_6_letter.arff");
		put("36", "dataset_36_segment.arff");
		put("38", "dataset_38_sick.arff");
		put("44", "dataset_44_spambase.arff");
		put("46", "dataset_46_splice.arff");
}};
	
	public static double[] getDatasetLandmarkers(SQLAdapter adapter, int dataset) throws SQLException {
		ResultSet res = adapter.getResultsOfQuery(
				"SELECT X_LANDMARKERS FROM " + DATASET_METAFEAT_TABLE + " WHERE dataset = " + dataset );
		
		Pattern arrayDeserializer = Pattern.compile(" ");
		res.first();	
		String serializedY = res.getString(1);
		double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble).toArray();
		
		return yArray;
	}
	
	public static double[] getInitialHyperPars(int numPars) {
		double[] initHyperPars = new double[numPars];
		for (int i = 0; i < numPars; i++) {
			initHyperPars[i] = 0.5;
		}
		
		return initHyperPars;
	}
	
	public static double evaluateJ48(INDArray hyperPars, Instances data, DyadNormalScaler scaler) {
		J48 j48 = new J48();
		double score = 0.0;
		
		double C = hyperPars.getDouble(0);
		double M = hyperPars.getDouble(1);
		
		// Undo normalization
		C *= scaler.getStatsY()[0].getMax() - scaler.getStatsY()[0].getMin();
		C += scaler.getStatsY()[0].getMin();
		
		M *= scaler.getStatsY()[1].getMax() - scaler.getStatsY()[1].getMin();
		M += scaler.getStatsY()[1].getMin();
		try {
			j48.setOptions(ZeroShotUtil.mapJ48InputsToWekaOptions(C, M));
		} catch (Exception e) {
			// Invalid parameters
			return 0.0;
		}
		try {
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(j48, data, 10, new Random(SEED));
			score = eval.pctCorrect();
		} catch (Exception e) {
			e.printStackTrace();
			return 0.0;
		}
		return score;
	}
	
	public static void main(String[] args) throws SQLException {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		File drDataFile = new File(DATA_PATH);
		DyadRankingDataset data = new DyadRankingDataset();
		
		try {
			data.deserialize(new FileInputStream(drDataFile));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		System.out.println("Data loaded");
		
		DyadNormalScaler scaler = new DyadNormalScaler();
		scaler.fit(data);
		PLNetDyadRanker plNet = new PLNetDyadRanker();
		try {
			plNet.loadModelFromFile(PLNET_PATH);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		for(int dataset : DATASETS_TEST) {
			System.out.println("Evaluating data set: " + dataset);
			
			double[] datasetFeatures = getDatasetLandmarkers(adapter, dataset);
			INDArray dsFeat = Nd4j.create(datasetFeatures);
			INDArray initHyperPars = Nd4j.create(getInitialHyperPars(NUM_HYPERPARS));
			INDArray inputMask = Nd4j.hstack(Nd4j.zeros(dsFeat.columns()), Nd4j.create(new double[]{1.0, 1.0}));			
			INDArray init = Nd4j.hstack(dsFeat, initHyperPars);
			
			int[] indicesToWatch = new int[NUM_HYPERPARS];
			for (int i = 0; i < NUM_HYPERPARS; i++) {
				indicesToWatch[i] = (int) (init.length() - NUM_HYPERPARS + i);
			}
			InputOptListener listener = new InputOptListener(indicesToWatch);
			PLNetInputOptimizer inputOpt = new PLNetInputOptimizer();
			inputOpt.setListener(listener);
			INDArray optimized = inputOpt.optimizeInput(
					plNet, init, new NegIdentityInpOptLoss(), LEARNING_RATE, NUM_ITERATIONS, inputMask);
			
			Instances evalData;
			try {
				evalData = new Instances(new BufferedReader(
						new FileReader(new File(EVAL_DATA_PATH 
								+ File.separator 
								+ datasetIdMap.get(dataset + "")))));
				evalData.setClassIndex(evalData.numAttributes() - 1);
				for (INDArray inp : listener.getInputList()) {
					System.out.print(evaluateJ48(inp, evalData, scaler) + ",");
				}
				System.out.println();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			for (double output : listener.getOutputList()) {
				System.out.print(output + ",");
			}
			System.out.println();
		}
	}
	
}
