package dyadranking.performance.mccvevaluation;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

import jaicore.basic.SQLAdapter;

public class LossUpdater {
	private static final String SUBSAMPLING_SIZE = "subsamplingSize";
	
	private static final String MCCV = "mccvSplit";
	
	private static final String TESTDATASET_ID = "testDatasetID";
	
	private static final String AVERAGING_RUN = "averagingRun";
	
	private static final String	K = "k";
	
	private static final String APPROACH_KEY = "approach";
		
	private static final String LOSS = "loss";
	
	private final String topKKTauTableName;
	
	private final String weverNumberAvgTableName;
	
	private final String weverNumberMinTableName;
	
	private final String kendallsTauTableName;
	
	private final String kendallsTauNormalizedTableName;
	
	private SQLAdapter adapter;
	
	public LossUpdater (SQLAdapter adapter, String tablePrefix) {
		this.adapter = adapter;
		topKKTauTableName = tablePrefix + "_topK_KendallsDistance";
		kendallsTauTableName = tablePrefix+"_KendallsTau";
		kendallsTauNormalizedTableName = tablePrefix+"_Normalized_KendallsTau";
		weverNumberMinTableName= tablePrefix+"_WeverNumber_Min";
		weverNumberAvgTableName = tablePrefix+"_WeverNumber_Avg";
	}

	public void updateTopKKTAULoss (int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun, int k, APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(K, Integer.toString(k));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		adapter.insert(topKKTauTableName, values);		
	}
	
	public void updateWeverNumberAvg (int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun, int k, APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(K, Integer.toString(k));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		adapter.insert(weverNumberAvgTableName, values);	
	}
	
	public void updateWeverNumberMin (int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun, int k, APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(K, Integer.toString(k));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		adapter.insert(weverNumberMinTableName, values);	
	}
	
	public void updateKendallsTau (int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun, APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		adapter.insert(kendallsTauTableName, values);	
	}
	
	public void updateKendallsTauAverage (int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun, APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		adapter.insert(kendallsTauNormalizedTableName, values);	
	}
}
