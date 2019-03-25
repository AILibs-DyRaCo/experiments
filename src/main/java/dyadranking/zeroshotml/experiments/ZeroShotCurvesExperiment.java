//package dyadranking.zeroshotml.experiments;
//
//import java.io.BufferedReader;
//import java.io.File;
//import java.io.FileInputStream;
//import java.io.FileNotFoundException;
//import java.io.FileReader;
//import java.io.IOException;
//import java.io.ObjectInputStream;
//import java.io.PrintWriter;
//import java.sql.ResultSet;
//import java.sql.SQLException;
//import java.util.HashMap;
//import java.util.Map;
//import java.util.Random;
//import java.util.concurrent.ExecutorService;
//import java.util.concurrent.Executors;
//import java.util.regex.Pattern;
//
//import org.aeonbits.owner.ConfigFactory;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.factory.Nd4j;
//
//import dyadranking.sql.SQLUtils;
//import jaicore.basic.SQLAdapter;
//import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
//import jaicore.ml.dyadranking.util.DyadNormalScaler;
//import jaicore.ml.dyadranking.zeroshot.inputoptimization.NegIdentityInpOptLoss;
//import jaicore.ml.dyadranking.zeroshot.inputoptimization.PLNetInputOptimizer;
//import jaicore.ml.dyadranking.zeroshot.util.InputOptListener;
//import jaicore.ml.dyadranking.zeroshot.util.ZeroShotUtil;
//import weka.classifiers.Evaluation;
//import weka.classifiers.functions.MultilayerPerceptron;
//import weka.classifiers.functions.SMO;
//import weka.classifiers.trees.J48;
//import weka.classifiers.trees.RandomForest;
//import weka.core.Instances;
//
//public class ZeroShotCurvesExperiment {
//	
//	private static IZeroShotCurvesConfig config = ConfigFactory.create(IZeroShotCurvesConfig.class);
//	
//	private static final String SCALER_PATH = config.getScalerPath();
//	
//	private static final String PLNET_PATH = config.getPLNetPath();
//	
//	private static final String EVAL_DATA_PATH = config.getEvalDataPath();
//	
//	private static final String OUTPUT_PATH = config.getOutputPath();
//	
//	private static final String DATASET_METAFEAT_TABLE = config.getMetaFeatTable();
//	
//	private static final int[] DATASETS_TEST = config.getTestDatasets();
//	
//	private static final String CLASSIFIER = config.getClassifier();
//
//	private static final double LEARNING_RATE = config.getLearningRate();
//
//	private static final int NUM_ITERATIONS = config.getNumIterations();
//	
//	private static final int SEED = config.getSeed();
//	
//	private static final double[] INIT_HYPERPARS = config.getInitHyperPars();
//	
//	private static final int RANDOM_RESTARTS = config.getRandomRestarts();
//	
//	private static final int RESTART_SEED = config.getRestartSeed();
//	
//	private static Map<String, String> datasetIdMap = new HashMap<String, String>() {{
//		put("12", "dataset_12_mfeat-factors.arff");
//		put("14", "dataset_14_mfeat-fourier.arff");
//		put("16", "dataset_16_mfeat-karhunen.arff");
//		put("18", "dataset_18_mfeat-morphological.arff");
//		put("20", "dataset_20_mfeat-pixel.arff");
//		put("21", "dataset_21_car.arff");
//		put("22", "dataset_22_mfeat-zernike.arff");
//		put("23", "dataset_23_cmc.arff");
//		put("24", "dataset_24_mushroom.arff");
//		put("26", "dataset_26_nursery.arff");
//		put("28", "dataset_28_optdigits.arff");
//		put("3", "dataset_3_kr-vs-kp.arff");
//		put("30", "dataset_30_page-blocks.arff");
//		put("32", "dataset_32_pendigits.arff");
//		put("5", "dataset_5_arrhythmia.arff");
//		put("6", "dataset_6_letter.arff");
//		put("36", "dataset_36_segment.arff");
//		put("38", "dataset_38_sick.arff");
//		put("44", "dataset_44_spambase.arff");
//		put("46", "dataset_46_splice.arff");
//		put("183", "dataset_183_abalone.arff");
//}};
//	
//	public static double[] getDatasetLandmarkers(SQLAdapter adapter, int dataset) throws SQLException {
//		ResultSet res = adapter.getResultsOfQuery(
//				"SELECT X_LANDMARKERS FROM " + DATASET_METAFEAT_TABLE + " WHERE dataset = " + dataset );
//		
//		Pattern arrayDeserializer = Pattern.compile(" ");
//		res.first();	
//		String serializedY = res.getString(1);
//		double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble).toArray();
//		
//		return yArray;
//	}
//	
//	public static double evaluateJ48(INDArray hyperPars, Instances data, DyadNormalScaler scaler) {
//		J48 j48 = new J48();
//		double score = 0.0;
//		
//		double C = hyperPars.getDouble(0);
//		double M = hyperPars.getDouble(1);
//		
//		// Undo normalization
//		C *= scaler.getStatsY()[0].getMax() - scaler.getStatsY()[0].getMin();
//		C += scaler.getStatsY()[0].getMin();		
//		M *= scaler.getStatsY()[1].getMax() - scaler.getStatsY()[1].getMin();
//		M += scaler.getStatsY()[1].getMin();
//		
//		try {
//			j48.setOptions(ZeroShotUtil.mapJ48InputsToWekaOptions(C, M));
//		} catch (Exception e) {
//			// Invalid parameters
//			return 0.0;
//		}
//		try {
//			Evaluation eval = new Evaluation(data);
//			eval.crossValidateModel(j48, data, 10, new Random(SEED));
//			score = eval.pctCorrect();
//		} catch (Exception e) {
//			e.printStackTrace();
//			return 0.0;
//		}
//		return score;
//	}
//	
//	public static double evaluateSMO(INDArray hyperPars, Instances data, DyadNormalScaler scaler) {
//		SMO smo = new SMO();
//		double score = 0.0;
//		
//		double CExp = hyperPars.getDouble(0);
//		double LExp = hyperPars.getDouble(1);
//		double RBFGammaExp = hyperPars.getDouble(2);
//		
//		// Undo normalization
//		CExp *= scaler.getStatsY()[0].getMax() - scaler.getStatsY()[0].getMin();
//		CExp += scaler.getStatsY()[0].getMin();		
//		LExp *= scaler.getStatsY()[1].getMax() - scaler.getStatsY()[1].getMin();
//		LExp += scaler.getStatsY()[1].getMin();		
//		RBFGammaExp *= scaler.getStatsY()[2].getMax() - scaler.getStatsY()[2].getMin();
//		RBFGammaExp += scaler.getStatsY()[2].getMin();
//		
//		try {
//			smo.setOptions(ZeroShotUtil.mapSMORBFInputsToWekaOptions(CExp, LExp, RBFGammaExp));
//		} catch (Exception e) {
//			// Invalid parameters
//			e.printStackTrace();
//			return 0.0;
//		}
//		try {
//			Evaluation eval = new Evaluation(data);
//			eval.crossValidateModel(smo, data, 5, new Random(SEED));
//			score = eval.pctCorrect();
//		} catch (Exception e) {
//			e.printStackTrace();
//			return 0.0;
//		}
//		
//		return score;
//	}
//	
//	public static double evaluateRF(INDArray hyperPars, Instances data, DyadNormalScaler scaler) {
//		RandomForest rf = new RandomForest();
//		double score = 0.0;
//		
//		double I = hyperPars.getDouble(0);
//		double K_fraction = hyperPars.getDouble(1);
//		double M = hyperPars.getDouble(2);
//		double depth = hyperPars.getDouble(3);
//		
//		// Undo normalization
//		I *= scaler.getStatsY()[0].getMax() - scaler.getStatsY()[0].getMin();
//		I += scaler.getStatsY()[0].getMin();		
//		K_fraction *= scaler.getStatsY()[1].getMax() - scaler.getStatsY()[1].getMin();
//		K_fraction += scaler.getStatsY()[1].getMin();		
//		M *= scaler.getStatsY()[2].getMax() - scaler.getStatsY()[2].getMin();
//		M += scaler.getStatsY()[2].getMin();
//		depth *= scaler.getStatsY()[3].getMax() - scaler.getStatsY()[3].getMin();
//		depth += scaler.getStatsY()[3].getMin();
//		
//		try {
//			rf.setOptions(ZeroShotUtil.mapRFInputsToWekaOptions(I, K_fraction, M, depth, data.numAttributes()));
//		} catch (Exception e) {
//			// Invalid parameters
//			e.printStackTrace();
//			return 0.0;
//		}
//		try {
//			Evaluation eval = new Evaluation(data);
//			eval.crossValidateModel(rf, data, 5, new Random(SEED));
//			score = eval.pctCorrect();
//		} catch (Exception e) {
//			e.printStackTrace();
//			return 0.0;
//		}
//		
//		return score;
//	}
//	
//	public static double evaluateMLP(INDArray hyperPars, Instances data, DyadNormalScaler scaler) {
//		MultilayerPerceptron mlp = new MultilayerPerceptron();
//		double score = 0.0;
//		
//		double L_exp = hyperPars.getDouble(0);
//		double M_exp = hyperPars.getDouble(1);
//		double N = hyperPars.getDouble(2);
//		
//		// Undo normalization
//		L_exp *= scaler.getStatsY()[0].getMax() - scaler.getStatsY()[0].getMin();
//		L_exp += scaler.getStatsY()[0].getMin();		
//		M_exp *= scaler.getStatsY()[1].getMax() - scaler.getStatsY()[1].getMin();
//		M_exp += scaler.getStatsY()[1].getMin();		
//		N *= scaler.getStatsY()[2].getMax() - scaler.getStatsY()[2].getMin();
//		N += scaler.getStatsY()[2].getMin();
//		
//		try {
//			mlp.setOptions(ZeroShotUtil.mapMLPInputsToWekaOptions(L_exp, M_exp, N));
//		} catch (Exception e) {
//			// Invalid parameters
//			e.printStackTrace();
//			return 0.0;
//		}
//		try {
//			Evaluation eval = new Evaluation(data);
//			eval.crossValidateModel(mlp, data, 5, new Random(SEED));
//			score = eval.pctCorrect();
//		} catch (Exception e) {
//			e.printStackTrace();
//			return 0.0;
//		}
//		
//		return score;
//	}
//	
//	public static void main(String[] args) throws SQLException, IOException {
//		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
//		
//		File scalerFile = new File(SCALER_PATH);
//		DyadNormalScaler scaler = null;
//		FileInputStream fileIn = null;
//		try {		
//			fileIn = new FileInputStream(scalerFile);
//			ObjectInputStream objIn = new ObjectInputStream(fileIn);
//			scaler = (DyadNormalScaler) objIn.readObject();
//			objIn.close();
//		} catch (ClassNotFoundException e) {
//			e.printStackTrace();
//			System.exit(1);
//		} catch (IOException e) {
//			e.printStackTrace();
//			System.exit(1);
//		} finally {
//			try {
//				fileIn.close();
//			} catch (IOException e) {
//				e.printStackTrace();
//				System.exit(1);
//			}
//		}		
//		DyadNormalScaler scalerFinal = scaler;
//		
//		int numHyperpars = 0;
//		switch(CLASSIFIER) {
//		case "j48":
//			numHyperpars = 2;
//			break;
//		case "rf":
//			numHyperpars = 4;
//			break;
//		case "smo":
//			numHyperpars = 3;
//			break;
//		case "mlp":
//			numHyperpars = 3;
//			break;
//		}
//		
//		int numHyperparsFinal = numHyperpars;
//		
//		ExecutorService executor = Executors.newFixedThreadPool(6);
//		
//		for(int dataset : DATASETS_TEST) {
//			executor.execute(() -> {
//			PLNetDyadRanker plNet = new PLNetDyadRanker();
//			try {
//				plNet.loadModelFromFile(PLNET_PATH);
//			} catch (IOException e) {
//				e.printStackTrace();
//				System.exit(1);
//			}		
//			
//			Random restartRng = new Random(RESTART_SEED);
//			int num_restarts = RANDOM_RESTARTS == 0 ? 1 : RANDOM_RESTARTS;
//			for (int rndStart = 0; rndStart < num_restarts; rndStart++) {
//				System.out.println("Evaluating data set: " + dataset + " try " + rndStart);
//				
//				File outputFile = new File(OUTPUT_PATH + dataset + "_" + rndStart);
//				if (!outputFile.exists()) {
//					try {
//						outputFile.createNewFile();
//					} catch (IOException e) {
//						// TODO Auto-generated catch block
//						e.printStackTrace();
//					}
//				}
//				PrintWriter outStream = null;
//				try {
//					outStream = new PrintWriter(outputFile);
//				} catch (FileNotFoundException e1) {
//					// TODO Auto-generated catch block
//					e1.printStackTrace();
//				}
//				
//				double[] initHyperParsArr = INIT_HYPERPARS;
//				if(RANDOM_RESTARTS > 0) {
//					initHyperParsArr = new double[numHyperparsFinal];
//					for (int i = 0; i < numHyperparsFinal; i++) {
//						initHyperParsArr[i] = restartRng.nextDouble();
//					}
//				}
//				
//				double[] datasetFeatures = null;
//				try {
//					datasetFeatures = getDatasetLandmarkers(adapter, dataset);
//				} catch (SQLException e1) {
//					// TODO Auto-generated catch block
//					e1.printStackTrace();
//				}
//				INDArray dsFeat = Nd4j.create(datasetFeatures);
//				INDArray initHyperPars = Nd4j.create(initHyperParsArr);
//				INDArray inputMask = Nd4j.hstack(Nd4j.zeros(dsFeat.columns()), Nd4j.ones(numHyperparsFinal));			
//				INDArray init = Nd4j.hstack(dsFeat, initHyperPars);
//				
//				int[] indicesToWatch = new int[numHyperparsFinal];
//				for (int i = 0; i < numHyperparsFinal; i++) {
//					indicesToWatch[i] = (int) (init.length() - numHyperparsFinal + i);
//				}
//				InputOptListener listener = new InputOptListener(indicesToWatch);
//				PLNetInputOptimizer inputOpt = new PLNetInputOptimizer();
//				inputOpt.setListener(listener);
//				INDArray optimized = inputOpt.optimizeInput(
//						plNet, init, new NegIdentityInpOptLoss(), LEARNING_RATE, NUM_ITERATIONS, inputMask);
//				
//				Instances evalData;
//				try {
//					evalData = new Instances(new BufferedReader(
//							new FileReader(new File(EVAL_DATA_PATH 
//									+ File.separator 
//									+ datasetIdMap.get(dataset + "")))));
//					evalData.setClassIndex(evalData.numAttributes() - 1);
//					for (INDArray inp : listener.getInputList()) {
//						double score = 0.0;
//						switch(CLASSIFIER) {
//						case "j48":
//							score = evaluateJ48(inp, evalData, scalerFinal);
//							break;
//						case "rf":
//							score = evaluateRF(inp, evalData, scalerFinal);
//							break;
//						case "smo":
//							score = evaluateSMO(inp, evalData, scalerFinal);
//							break;
//						case "mlp":
//							score = evaluateMLP(inp, evalData, scalerFinal);
//							break;
//						}
//						outStream.print(score + ",");
//					}
//					outStream.println();
//				} catch (FileNotFoundException e) {
//					// TODO Auto-generated catch block
//					e.printStackTrace();
//				} catch (IOException e) {
//					// TODO Auto-generated catch block
//					e.printStackTrace();
//				}
//				for (double output : listener.getOutputList()) {
//					outStream.print(output + ",");
//				}
//				outStream.println();
//				outStream.println(ZeroShotUtil.unscaleParameters(optimized, scalerFinal, numHyperparsFinal));
//				outStream.close();
//			}
//		});
//		}
//	}
//	
//}
