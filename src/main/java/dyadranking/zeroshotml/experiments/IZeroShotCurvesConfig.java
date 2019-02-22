package dyadranking.zeroshotml.experiments;

import org.aeonbits.owner.Mutable;
import org.aeonbits.owner.Config.Sources;

@Sources({ "file:conf/zeroshotexp/zeroshot_curves.properties" })
public interface IZeroShotCurvesConfig extends Mutable {
	public static final String SCALER_PATH = "scalerpath";
	
	public static final String PLNET_PATH = "plnetpath";
	
	public static final String EVAL_DATA_PATH = "evalpath";
	
	public static final String OUTPUT_PATH = "outputpath";
	
	public static final String DATASET_METAFEAT_TABLE = "datasetmetafeaturestable";
	
	public static final String DATASETS_TEST = "datasetstest";
	
	public static final String CLASSIFIER = "classifier";

	public static final String LEARNING_RATE = "learningrate";

	public static final String NUM_ITERATIONS = "numiterations";
	
	public static final String SEED = "seed";
	
	public static final String INIT_HYPERPARS = "init_hyperpars";
	
	public static final String RANDOM_RESTARTS = "randomrestarts";
	
	public static final String RESTART_SEED = "restartseed";
	
	@Key(SCALER_PATH)
	public String getScalerPath();
	
	@Key(PLNET_PATH)
	public String getPLNetPath();
	
	@Key(EVAL_DATA_PATH)
	public String getEvalDataPath();
	
	@Key(OUTPUT_PATH)
	public String getOutputPath();
	
	@Key(DATASET_METAFEAT_TABLE)
	public String getMetaFeatTable();
	
	@Key(DATASETS_TEST)
	public int[] getTestDatasets();
	
	@Key(CLASSIFIER)
	public String getClassifier();
	
	@Key(LEARNING_RATE)
	public double getLearningRate();
	
	@Key(NUM_ITERATIONS)
	public int getNumIterations();
	
	@Key(SEED)
	public int getSeed();
	
	@Key(INIT_HYPERPARS)
	public double[] getInitHyperPars();
	
	@Key(RANDOM_RESTARTS)
	public int getRandomRestarts();
	
	@Key(RESTART_SEED)
	public int getRestartSeed();
}
