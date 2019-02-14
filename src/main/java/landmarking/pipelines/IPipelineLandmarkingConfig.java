package landmarking.pipelines;

import java.util.List;

import org.aeonbits.owner.Config.DefaultValue;
import org.aeonbits.owner.Config.Key;
import org.aeonbits.owner.Config.Sources;

import jaicore.experiments.IExperimentSetConfig;

/**
 * In fact, this config is an exact copy of IMultiClassClassificationExperimentConfig,
 * which cannot be used here to avoid cyclic dependencies.
 * Typically, in this case, you just would extend IMultiClassClassificationExperimentConfig
 * and leave the interface (mostly) empty except that you define the URL for the file.
 * 
 * 
 * @author fmohr
 *
 */
@Sources({ "file:./examples/mlexample/setup.properties" })
public interface IPipelineLandmarkingConfig extends IExperimentSetConfig {
	
	/**
	 * Number of repetitions for monte carlo cross validation.
	 */
	public static final String K_MCCV_REPEATS = "mccvRepeats";
	
	/**
	 * Seed for monte carlo cross validation.
	 */
	public static final String K_MCCV_SEED = "mccvSeed";
	
	/**
	 * Ratio of data used for training in monte carlo cross validation.
	 */
	public static final String K_MCCV_TRAIN_RATIO = "mccvTrainRatio";
	
	/**
	 * OpenML Key.
	 */
	public static final String K_OPENML_KEY = "openMLKey";
	
	/**
	 * OpenML dataset IDs.
	 */
	public static final String DATASETS_IDS = "dataset_ids";
	
	@Key(K_MCCV_REPEATS)
	@DefaultValue("10")
	public int getNumberMCCVRepeats();

	@Key(K_MCCV_SEED)
	@DefaultValue("1")
	public int getMCCVSeed();

	@Key(K_MCCV_TRAIN_RATIO)
	@DefaultValue("0.7d")
	public double getMCCVTrainRatio();

	@Key(K_RANDOM_SEARCH_SEED)
	@DefaultValue("1")
	public int getRandomSearchSeed();
	
	@Key(K_OPENML_KEY)
	public String getOpenMLKey();

	@Key(DATASETS_IDS)
	public List<String> getDatasetIDs();

}
