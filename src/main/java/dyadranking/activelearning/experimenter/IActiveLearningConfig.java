package dyadranking.activelearning.experimenter;

import java.io.File;
import java.util.List;

import org.aeonbits.owner.Config.Sources;

import jaicore.experiments.IExperimentSetConfig;

@Sources({ "classpath:./dyadranking/activelearning/experimenter/setup.properties" })
public interface IActiveLearningConfig extends IExperimentSetConfig {
	public static final String SAMPLING_STRATEGIES = "samplingstrategies";
	public static final String SEEDS = "seeds";
	public static final String REMOVE_QUERIED_DYADS = "removeQueriedDyads";
	public static final String POOLS = "pools";
	public static final String NUMBER_QUERIES = "number_queries";
	public static final String TRAIN_RATIO = "train_ratios";
	
	@Key(SAMPLING_STRATEGIES)
	public List<String> getSamplingStrategies();
	
	@Key(POOLS)
	public List<String> getPools();
	
	@Key(SEEDS)
	public List<String> getSeeds();
	
	@Key(REMOVE_QUERIED_DYADS)
	public List<String> getRemoveQueriedDyads();

	@Key(NUMBER_QUERIES)
	public List<String> getNumberQueries();
	
	@Key(TRAIN_RATIO)
	public List<String> getTrainRatios();

}
