package dyadranking.performance;

import java.util.List;

import org.aeonbits.owner.Config.Sources;
import org.aeonbits.owner.Mutable;

@Sources({ "file:conf/dyad_performance/dyad_dataset_generator.properties" })
public interface DyadDatasetGeneratorConfig extends Mutable {

	public static final String SEED_KEY = "seeds";

	public static final String DATASETS_KEY = "dataset_ids";

	public static final String RANKING_TRAIN_KEY = "ranking_length_train";

	public static final String RANKING_TEST_KEY = "ranking_length_test";

	public static final String RANKING_NUM_TRAIN = "ranking_num_train";
	
	public static final String TABLE_KEY = "dyad_table";

	public static final String USE_ALL_DATA_KEY = "use_all_data";

	@Key(RANKING_NUM_TRAIN)
	public List<Integer> getRankingNumTrain();
	
	@Key(USE_ALL_DATA_KEY)
	public boolean useAllData();

	@Key(TABLE_KEY)
	public String getTableName();

	@Key(RANKING_TRAIN_KEY)
	public List<Integer> getRankingLengthsTrainKey();

	@Key(RANKING_TEST_KEY)
	public List<Integer> getRankingLengthsTest();

	@Key(DATASETS_KEY)
	public List<Integer> getDatasetIds();

	@Key(SEED_KEY)
	public List<Integer> getSeeds();
}
