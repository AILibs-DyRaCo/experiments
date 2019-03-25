//package dyadranking.zeroshotml.experiments;
//
//import java.io.File;
//import java.io.FileInputStream;
//import java.io.FileNotFoundException;
//import java.io.IOException;
//import java.io.ObjectInputStream;
//import java.io.PrintWriter;
//import java.sql.ResultSet;
//import java.sql.SQLException;
//import java.util.Random;
//import java.util.regex.Pattern;
//
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
//
//public class ZeroShotRawCurvesGenerator {
//	
//	private static final String SCALER_PATH = "datasets/zeroshot/RFtrain.dr_scaler";
//	
//	private static final String PLNET_PATH = "datasets/zeroshot/RFPLNet.plnet.zip";
//	
//	private static final String OUTPUT_PATH = "datasets/zeroshot/eval/rf_random_0.003-0.001";
//	
//	private static final String DATASET_METAFEAT_TABLE = "`dataset_metafeatures_mirror`";
//	
//	private static final int[] DATASETS_TEST = new int[] { 5 };
//	
//	private static final double LEARNING_RATE_INIT = 0.003;
//	
//	private static final double LEARNING_RATE_FINAL = 0.001;
//
//	private static final int NUM_ITERATIONS = 200;
//	
//	//private static final double[] INIT_HYPERPARS = new double[] { 0.477, 0.301, 0.246 };
//	private static final double[] INIT_HYPERPARS = null;
//	
//	private static final int NUM_HYPERPARS = 4;
//	
//	private static final int[] SEEDS = new int[] { 1, 2, 3, 4, 5};
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
//	@SuppressWarnings("unused")
//	public static void main(String[] args) {
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
//		
//		for (int dataset : DATASETS_TEST) {
//			for (int seed : SEEDS) {
//				PLNetDyadRanker plNet = new PLNetDyadRanker();
//				try {
//					plNet.loadModelFromFile(PLNET_PATH);
//				} catch (IOException e) {
//					e.printStackTrace();
//					System.exit(1);
//				}					
//				System.out.println("Evaluating data set: " + dataset);				
//	
//				/* Initialize plNet inputs */
//				double[] initHyperParsArr = INIT_HYPERPARS;
//				if(initHyperParsArr == null) {
//					Random rng = new Random(seed);
//					initHyperParsArr = new double[NUM_HYPERPARS];
//					for (int i = 0; i < NUM_HYPERPARS; i++) {
//						initHyperParsArr[i] = rng.nextDouble();
//					}
//				}				
//				double[] datasetFeatures = null;
//				try {
//					datasetFeatures = getDatasetLandmarkers(adapter, dataset);
//				} catch (SQLException e1) {
//					// TODO Auto-generated catch block
//					e1.printStackTrace();
//				}				
//				INDArray dsFeat = Nd4j.create(datasetFeatures);
//				INDArray initHyperPars = Nd4j.create(initHyperParsArr);
//				INDArray inputMask = Nd4j.hstack(Nd4j.zeros(dsFeat.columns()), Nd4j.ones(NUM_HYPERPARS));			
//				INDArray init = Nd4j.hstack(dsFeat, initHyperPars);
//	
//				/* Optimize plNet input */
//				int[] indicesToWatch = new int[NUM_HYPERPARS];
//				for (int i = 0; i < NUM_HYPERPARS; i++) {
//					indicesToWatch[i] = (int) (init.length() - NUM_HYPERPARS + i);
//				}
//				InputOptListener listener = new InputOptListener(indicesToWatch);
//				PLNetInputOptimizer inputOpt = new PLNetInputOptimizer();
//				inputOpt.setListener(listener);
//				INDArray optimized = inputOpt.optimizeInput(
//						plNet, init, new NegIdentityInpOptLoss(), LEARNING_RATE_INIT, LEARNING_RATE_FINAL, NUM_ITERATIONS, inputMask);
//	
//				/* Write optimization progress to file */
//				String outputPath = OUTPUT_PATH + "_" + dataset + "_" + seed;
//				File parsOutputFile = new File(outputPath + ".rawpars");
//				if (!parsOutputFile.exists()) {
//					try {
//						parsOutputFile.createNewFile();
//					} catch (IOException e) {
//						// TODO Auto-generated catch block
//						e.printStackTrace();
//					}
//				}
//				PrintWriter parsOutStream = null;
//				try {
//					parsOutStream = new PrintWriter(parsOutputFile);
//				} catch (FileNotFoundException e1) {
//					// TODO Auto-generated catch block
//					e1.printStackTrace();
//				}
//	
//				for (INDArray pars : listener.getInputList()) {
//					INDArray unscaledPars = ZeroShotUtil.unscaleParameters(pars, scaler, NUM_HYPERPARS);
//					for (int i = 0; i < unscaledPars.length() - 1; i++) {
//						parsOutStream.print(unscaledPars.getDouble(i) + ",");
//					}
//					parsOutStream.println(unscaledPars.getDouble(unscaledPars.length() - 1));
//				}
//				parsOutStream.close();
//	
//				File outputsOutputFile = new File(outputPath + ".outputs");
//				if (!outputsOutputFile.exists()) {
//					try {
//						outputsOutputFile.createNewFile();
//					} catch (IOException e) {
//						// TODO Auto-generated catch block
//						e.printStackTrace();
//					}
//				}
//				PrintWriter outputsOutStream = null;
//				try {
//					outputsOutStream = new PrintWriter(outputsOutputFile);
//				} catch (FileNotFoundException e1) {
//					// TODO Auto-generated catch block
//					e1.printStackTrace();
//				}
//				for (Double output : listener.getOutputList()) {
//					outputsOutStream.println(output);
//				}
//				outputsOutStream.close();
//			}
//		}
//	}
//}
