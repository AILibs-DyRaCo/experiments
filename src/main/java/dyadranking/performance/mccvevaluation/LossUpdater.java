package dyadranking.performance.mccvevaluation;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import jaicore.basic.SQLAdapter;

public class LossUpdater {
	private static final String SUBSAMPLING_SIZE = "subsamplingSize";

	private static final String MCCV = "mccvSplit";

	private static final String TESTDATASET_ID = "testDatasetID";

	private static final String AVERAGING_RUN = "averagingRun";

	private static final String K = "k";

	private static final String APPROACH_KEY = "approach";

	private static final String LOSS = "loss";

	private final String topKKTauTableName;

	private final String weverNumberAvgTableName;

	private final String weverNumberMinTableName;

	private final String kendallsTauTableName;

	private final String kendallsTauNormalizedTableName;

	private SQLAdapter adapter;

	public LossUpdater(SQLAdapter adapter, String tablePrefix) {
		this.adapter = adapter;
		topKKTauTableName = tablePrefix + "_topK_KendallsDistance";
		kendallsTauTableName = tablePrefix + "_KendallsTau";
		kendallsTauNormalizedTableName = tablePrefix + "_Normalized_KendallsTau";
		weverNumberMinTableName = tablePrefix + "_WeverNumber_Min";
		weverNumberAvgTableName = tablePrefix + "_WeverNumber_Avg";
		try {
			createTables();
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}

	public void updateTopKKTAULoss(int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun, int k,
			APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(K, Integer.toString(k));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		try {
			adapter.insert(topKKTauTableName, values);
		} catch (SQLException e) {
		}
	}

	public void updateWeverNumberAvg(int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun,
			int k, APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(K, Integer.toString(k));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		try {
			adapter.insert(weverNumberAvgTableName, values);
		} catch (SQLException e) {
		}

	}

	public void updateWeverNumberMin(int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun,
			int k, APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(K, Integer.toString(k));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		try {
			adapter.insert(weverNumberMinTableName, values);
		} catch (SQLException e) {
		}

	}

	public void updateKendallsTau(int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun,
			APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		try {
			adapter.insert(kendallsTauTableName, values);
		} catch (SQLException e) {
		}

	}

	public void updateKendallsTauAverage(int subSamplingSize, int mccvSplit, int testDatasetId, int innerAveragingRun,
			APPROACH approach, double loss) throws SQLException {
		Map<String, String> values = new HashMap<>();
		values.put(SUBSAMPLING_SIZE, Integer.toString(subSamplingSize));
		values.put(MCCV, Integer.toString(mccvSplit));
		values.put(TESTDATASET_ID, Integer.toString(testDatasetId));
		values.put(AVERAGING_RUN, Integer.toString(innerAveragingRun));
		values.put(APPROACH_KEY, approach.toString());
		values.put(LOSS, Double.toString(loss));
		try {
			adapter.insert(kendallsTauNormalizedTableName, values);
		} catch (SQLException e) {
		}
	}

	private void createTables() throws SQLException {
		createNormalTable(kendallsTauNormalizedTableName);
		createNormalTable(kendallsTauTableName);
		createKTable(weverNumberAvgTableName);
		createKTable(weverNumberMinTableName);
		createKTable(topKKTauTableName);
	}

	private void createKTable(String tableName) throws SQLException {
		ResultSet rs = adapter.getResultsOfQuery("SHOW TABLES");
		boolean hasPerformanceTable = false;
		while (rs.next()) {
			String ptableName = rs.getString(1);
			if (ptableName.equals(tableName))
				hasPerformanceTable = true;
		}
		if (hasPerformanceTable) {
			//adapter.update("DELETE FROM " + tableName + " WHERE 1=1");
		} else {

			adapter.update("CREATE TABLE " + tableName + " (`" + K + "` int(10) NOT NULL, \r \n" + " `"
					+ SUBSAMPLING_SIZE + "` int(10) NOT NULL,\r\n" + " `" + MCCV + "` int(10) NOT NULL,\r\n" + " `"
					+ TESTDATASET_ID + "` int(10) NOT NULL, \r \n" + " `" + AVERAGING_RUN + "` int(10) NOT NULL, \r \n"
					+ " `" + LOSS + "` double NOT NULL, \r \n" + " `" + APPROACH_KEY
					+ "` VARCHAR(22) NOT NULL \r \n) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin",
					new ArrayList<>());
		}
	}

	private void createNormalTable(String tableName) throws SQLException {
		ResultSet rs = adapter.getResultsOfQuery("SHOW TABLES");
		boolean hasPerformanceTable = false;
		while (rs.next()) {
			String ptableName = rs.getString(1);
			if (ptableName.equals(tableName))
				hasPerformanceTable = true;
		}
		if (hasPerformanceTable) {
		//	adapter.update("DELETE FROM " + tableName + " WHERE 1=1");
		} else {

			adapter.update("CREATE TABLE " + tableName + " (`" + SUBSAMPLING_SIZE + "` int(10) NOT NULL, \r \n" + " `"
					+ MCCV + "` int(10) NOT NULL,\r\n" + " `" + TESTDATASET_ID + "` int(10) NOT NULL, \r \n" + " `"
					+ AVERAGING_RUN + "` int(10) NOT NULL, \r \n" + " `" + LOSS + "` double NOT NULL, \r \n" + " `"
					+ APPROACH_KEY + "` VARCHAR(22) NOT NULL\r \n) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin",
					new ArrayList<>());
		}
	}
}
