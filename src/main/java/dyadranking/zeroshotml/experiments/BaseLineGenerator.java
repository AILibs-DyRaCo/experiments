package dyadranking.zeroshotml.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class BaseLineGenerator {
	
	private static final String DATA_PATH = "datasets/test";
	
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
	
	private static final int[] DATASETS_TEST = { 5, 6, 36, 38, 44, 46 };
	
	private static final int SEED = 42;
	
	public static void main(String[] args) throws Exception {

		for(int dataset : DATASETS_TEST) {
			System.out.println("Data set " + dataset + " default performances:");
			J48 j48 = new J48();
			SMO smo = new SMO();
			RandomForest rf = new RandomForest();
			MultilayerPerceptron mlp = new MultilayerPerceptron();
			
			smo.setKernel(new RBFKernel());
			
			Instances data = null;
			BufferedReader dataReader = null;
			try {
				dataReader = new BufferedReader(
						new FileReader(new File(DATA_PATH 
								+ File.separator 
								+ datasetIdMap.get(dataset + ""))));
				data = new Instances(dataReader);
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				try {
					dataReader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			data.setClassIndex(data.numAttributes() - 1);
			//Evaluation j48Eval = new Evaluation(data);
			//Evaluation smoEval = new Evaluation(data);
			//Evaluation rfEval = new Evaluation(data);
			Evaluation mlpEval = new Evaluation(data);
			//j48Eval.crossValidateModel(j48, data, 10, new Random(SEED));
			//smoEval.crossValidateModel(smo, data, 5, new Random(SEED));
			//rfEval.crossValidateModel(rf, data, 5, new Random(SEED));
			mlpEval.crossValidateModel(mlp, data, 5, new Random(SEED));
			//System.out.println("J48 : " + j48Eval.pctCorrect());
			//System.out.println("SMO : " + smoEval.pctCorrect());
			//System.out.println("RF : " + rfEval.pctCorrect());
			System.out.println("MLP : " + mlpEval.pctCorrect());
			System.out.println();
			
		}

	}

}
