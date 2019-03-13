package dyadranking.activelearning.experimenter;

import java.io.File;
import java.util.List;

import org.aeonbits.owner.Config.Key;
import org.aeonbits.owner.Config.Sources;

import jaicore.experiments.IExperimentSetConfig;

@Sources({ "file:./activelearner.properties" })
public interface IActiveLearningConfig extends IExperimentSetConfig {
	public static final String SAMPLING_STRATEGIES = "sampling_strategies";
	public static final String SEEDS = "seeds";
	public static final String REMOVE_QUERIED_DYADS = "remove_queried_dyads";
	public static final String NUMBER_QUERIES = "number_queries";
	public static final String TRAIN_RATIO = "train_ratios";
	public static final String MINIBATCH_RATIO_OF_OLD_INSTANCES = "minibatch_ratio_of_old_instances";
	public static final String MINIBATCH_SIZE = "minibatch_sizes";
	public static final String DATASETFOLDER = "datasetfolder";
	public static final String DATASETS = "datasets";
	public static final String LENGTH_OF_TOP_RANKING = "length_of_top_rankings";
	public static final String LEARNING_CURVE_TABLENAME = "learning_curve_tablename";
	public static final String MEASURES = "measures";
	public static final String NUMBER_RANDOM_QUERIES_AT_START = "number_random_queries_at_start";
	public static final String TRANSFORM_INSTANCES = "transform_instances";
	public static final String TRANSFORM_ALTERNATIVES = "transform_alternatives";
	public static final String SCALING_METHOD = "scaling_method";

	
	@Key(SAMPLING_STRATEGIES)
	public List<String> getSamplingStrategies();
	
	@Key(SEEDS)
	public List<String> getSeeds();
	
	@Key(REMOVE_QUERIED_DYADS)
	@DefaultValue("true")
	public List<String> getRemoveQueriedDyads();

	@Key(NUMBER_QUERIES)
	public List<String> getNumberQueries();
	
	@Key(TRAIN_RATIO)
	public String getTrainRatios();

	@Key(MINIBATCH_RATIO_OF_OLD_INSTANCES)
	public String getMinibatchRatioOfOldInstances();

	@Key(MINIBATCH_SIZE)
	public List<String> getMinibatchSize();

	@Key(DATASETFOLDER)
	public File getDatasetFolder();
	
	@Key(LEARNING_CURVE_TABLENAME)
	public String getLearningCurveTableName();
	
	@Key(DATASETS)
	public List<String> getDatasets();
	
	@Key(LENGTH_OF_TOP_RANKING)
	public String getLengthOfTopRanking();
	
	@Key(MEASURES)
	public List<String> getMeasures();
	
	@Key(NUMBER_RANDOM_QUERIES_AT_START)
	public String getNumberRandomQueriesAtStart();
	
	@Key(NUMBER_RANDOM_QUERIES_AT_START)
	public String getTransformInstances();
	
	@Key(NUMBER_RANDOM_QUERIES_AT_START)
	public String getTransformAlternatives();
	
	@Key(NUMBER_RANDOM_QUERIES_AT_START)
	public String getScalingMethod();
}
