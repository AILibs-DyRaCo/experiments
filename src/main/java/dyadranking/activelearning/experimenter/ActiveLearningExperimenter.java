package dyadranking.activelearning.experimenter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.primitives.Pair;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.activelearning.ActiveDyadRanker;
import jaicore.ml.dyadranking.activelearning.DyadScorePoolProvider;
import jaicore.ml.dyadranking.activelearning.PrototypicalPoolBasedActiveDyadRanker;
import jaicore.ml.dyadranking.activelearning.RandomPoolBasedActiveDyadRanker;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;

public class ActiveLearningExperimenter {

	private static String DYADS_SCORES_FILE = "./meta-mining-dyads.txt";
	private static String LEARNING_CURVE_TABLE = "active_ranking_learning_curves";

	private static List<Pair<Dyad, Double>> loadDyadsAndScore(String filePath) {
		List<Pair<Dyad, Double>> dyadScorePairs = new LinkedList<Pair<Dyad, Double>>();
		try {
			FileInputStream in = new FileInputStream(new File(filePath));
			String input = IOUtils.toString(in, StandardCharsets.UTF_8);
			String[] rows = input.split("\n");
			for (String row : rows) {
				if (row.isEmpty())
					break;
				List<Dyad> dyads = new LinkedList<Dyad>();
				String[] dyadTokens = row.split("\\|");
				String dyadString = dyadTokens[0];
				String[] values = dyadString.split(";");
				if (values[0].length() > 1 && values[1].length() > 1) {
					String[] instanceValues = values[0].substring(1, values[0].length() - 1).split(",");
					String[] alternativeValues = values[1].substring(1, values[1].length() - 1).split(",");
					Vector instance = new DenseDoubleVector(instanceValues.length);
					for (int i = 0; i < instanceValues.length; i++) {
						instance.setValue(i, Double.parseDouble(instanceValues[i]));
					}

					Vector alternative = new DenseDoubleVector(alternativeValues.length);
					for (int i = 0; i < alternativeValues.length; i++) {
						alternative.setValue(i, Double.parseDouble(alternativeValues[i]));
					}
					Dyad dyad = new Dyad(instance, alternative);

					Double score = Double.parseDouble(dyadTokens[1]);

					dyadScorePairs.add(new Pair<Dyad, Double>(dyad, score));
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return dyadScorePairs;
	}

	public static void main(final String[] args) {

		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		PLNetDyadRanker ranker = new PLNetDyadRanker();

		for (int seed = 1; seed < 30; seed++) {
			System.out.println();
			/* get experiment setup */
			String poolName = "automl";
			String samplingStrategy = "prototypical";
			boolean removeQueriedDyads = true;
			double trainRatio = 0.7;
			int numberQueries = 100;

			List<Pair<Dyad, Double>> dyadScorePairs = loadDyadsAndScore(DYADS_SCORES_FILE);

			DyadScorePoolProvider poolProvider = new DyadScorePoolProvider(dyadScorePairs);

			poolProvider.setRemoveDyadsWhenQueried(true);
			DyadRankingDataset dataset = new DyadRankingDataset();
			for (Vector vector : poolProvider.getInstanceFeatures()) {
				dataset.add(poolProvider.getDyadRankingInstanceForInstanceFeatures(vector));
			}

			Collections.shuffle(dataset, new Random(seed));

			DyadRankingDataset trainData = new DyadRankingDataset(
					dataset.subList(0, (int) (trainRatio * dataset.size())));
			DyadRankingDataset testData = new DyadRankingDataset(
					dataset.subList((int) (trainRatio * dataset.size()), dataset.size()));

			System.out.println("size before: " + poolProvider.getInstanceFeatures().size());

			System.out.println("train data: ");
			for (IInstance instance : trainData) {
				IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
				System.out.println(drInstance.getDyadAtPosition(0).getInstance());

			}

			System.out.println("test data: ");
			for (IInstance instance : testData) {
				IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
				System.out.println(drInstance.getDyadAtPosition(0).getInstance());
			}

			for (IInstance instance : testData) {
				IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;

				poolProvider.removeDyadsFromPoolByInstances(drInstance.getDyadAtPosition(0).getInstance());

			}

			System.out.println("size after: " + poolProvider.getInstanceFeatures().size());

//				ActiveDyadRanker activeRanker = new PrototypicalPoolBasedActiveDyadRanker(ranker, poolProvider);
			ActiveDyadRanker activeDyadRanker;
			if (samplingStrategy.equals("random"))
				activeDyadRanker = new RandomPoolBasedActiveDyadRanker(ranker, poolProvider, seed);
			else
				activeDyadRanker = new PrototypicalPoolBasedActiveDyadRanker(ranker, poolProvider);

			/* initialize tables if not existent */
//			try {
//				ResultSet rs = adapter.getResultsOfQuery("SHOW TABLES");
//				boolean hasPerformanceTable = false;
//				while (rs.next()) {
//					String tableName = rs.getString(1);
//					System.out.println(tableName);
//					if (tableName.equals(LEARNING_CURVE_TABLE)) {
//						hasPerformanceTable = true;
//						System.out.println("table already there");
//					}
//				}
//
//				// if there is no performance table, create it. we hash the composition and
//				// trajectory and use the hash value as primary key for performance reasons.
//				if (!hasPerformanceTable) {
//					System.out.println("creating table");
//					adapter.update("CREATE TABLE `" + LEARNING_CURVE_TABLE + "` (\r\n"
//							+ " `evaluation_id` int(10) NOT NULL AUTO_INCREMENT,\r\n"
//							+ " `sampling_strat` VARCHAR(30),\r\n" + " `query` int(10) NOT NULL,\r\n"
//							+ " `seed` int(10) NOT NULL,\r\n" + " `kendall_tau` double NOT NULL,\r\n"
//							+ " `pool` VARCHAR(30) NOT NULL,\r\n" + "`evaluation_date` timestamp NULL DEFAULT NULL,"
//							+ " PRIMARY KEY (`evaluation_id`)\r\n"
//							+ ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COLLATE=utf8_bin",
//							new ArrayList<>());
//				}
//
//			} catch (SQLException e) {
//				e.printStackTrace();
//			}

			try {

				// train the ranker
				for (int i = 0; i < 100; i++) {
					activeDyadRanker.activelyTrain(1);
					double avgKendallTau = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(),
							testData, ranker);
					System.out.print(avgKendallTau + ",");
//						Map<String, String> valueMap = new HashMap<>();
//						valueMap.put("sampling_strat", samplingStrategy);
//						valueMap.put("query", ""+i);
//						valueMap.put("seed", ""+seed);
//						valueMap.put("kendall_tau", ""+avgKendallTau);
//						valueMap.put("pool", poolName);
//						valueMap.put("evaluation_date",
//								new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(Date.from(Instant.now())));
//						adapter.insert(LEARNING_CURVE_TABLE, valueMap);
				}

			} catch (Exception e) {
				e.printStackTrace();
			}

		}
	}

}
