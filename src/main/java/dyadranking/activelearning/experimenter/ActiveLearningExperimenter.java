package dyadranking.activelearning.experimenter;

import java.io.File;
import java.io.FileInputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.aeonbits.owner.ConfigCache;

import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.dyadranking.activelearning.ActiveDyadRanker;
import jaicore.ml.dyadranking.activelearning.DyadDatasetPoolProvider;
import jaicore.ml.dyadranking.activelearning.PrototypicalPoolBasedActiveDyadRanker;
import jaicore.ml.dyadranking.activelearning.RandomPoolBasedActiveDyadRanker;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;

public class ActiveLearningExperimenter {

	public static void main(final String[] args) {
		IActiveLearningConfig m = ConfigCache.getOrCreate(IActiveLearningConfig.class);
		if (m.getDatasetFolder() == null || !m.getDatasetFolder().exists()) {
			throw new IllegalArgumentException("config specifies invalid dataset folder " + m.getDatasetFolder());
		}

		ExperimentRunner runner = new ExperimentRunner(new IExperimentSetEvaluator() {

			@Override
			public IExperimentSetConfig getConfig() {
				return m;
			}

			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final SQLAdapter adapter,
					final IExperimentIntermediateResultProcessor processor) throws Exception {

				/* get experiment setup */
				Map<String, String> description = experimentEntry.getExperiment().getValuesOfKeyFields();
				int seed = Integer.parseInt(description.get("seed"));
				String datasetName = description.get(IActiveLearningConfig.DATASETS);
				int minibatchSize = Integer.parseInt(description.get("minibatch_size"));
				double ratioOfOldInstancesInMinibatch = Double
						.parseDouble(description.get("minibatch_ratio_of_old_instance"));
				boolean removeQueriedDyadsFromPool = Boolean.parseBoolean(description.get("remove_queried_dyad"));
				int numberQueries = Integer.parseInt(description.get("number_queries"));
				String samplingStrategy = description.get("sampling_strategie");
				double trainRatio = Double.parseDouble(description.get("train_ratio"));
				int lengthOfTopRankingToConsider = Integer.parseInt(description.get("length_of_top_ranking"));
				String curveTable = m.getLearningCurveTableName();

				/* initialize learning curve table if not existent */
				try {
					ResultSet rs = adapter.getResultsOfQuery("SHOW TABLES");
					boolean hasPerformanceTable = false;
					while (rs.next()) {
						String tableName = rs.getString(1);
						if (tableName.equals(curveTable))
							hasPerformanceTable = true;
					}

					if (!hasPerformanceTable) {
						adapter.update("CREATE TABLE `" + curveTable + "` (\r\n"
								+ " `evaluation_id` int(10) NOT NULL AUTO_INCREMENT,\r\n" + " `seed` json NOT NULL,\r\n"
								+ " `query_step` int NOT NULL,\r\n" + "`sampling_strategy` varchar(200) NOT NULL,\r\n"
								+ " `train_ratio` double NOT NULL,\r\n"
								+ " `old_data_in_minibatch_ratio` double NOT NULL,\r\n"
								+ " `minibatch_size` int NOT NULL,\r\n" + " `dataset` varchar(200) NOT NULL,\r\n"
								+ " `remove_queried_dyads` bit ,\r\n" + " `measure` varchar(200) NOT NULL,\r\n"
								+ " `score` double NOT NULL,\r\n" + "`evaluation_date` timestamp NULL DEFAULT NULL,"
								+ " PRIMARY KEY (`evaluation_id`)\r\n"
								+ ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COLLATE=utf8_bin",
								new ArrayList<>());
					}

				} catch (SQLException e) {
					e.printStackTrace();
				}

				DyadRankingDataset dataset = new DyadRankingDataset();
				dataset.deserialize(new FileInputStream(new File(m.getDatasetFolder() + datasetName)));
				Collections.shuffle(dataset, new Random(seed));
				DyadRankingDataset trainData = new DyadRankingDataset(
						dataset.subList(0, (int) (dataset.size() * trainRatio)));
				DyadRankingDataset testData = new DyadRankingDataset(
						dataset.subList((int) (dataset.size() * trainRatio), dataset.size()));
				DyadDatasetPoolProvider poolProvider = new DyadDatasetPoolProvider(trainData);
				poolProvider.setRemoveDyadsWhenQueried(removeQueriedDyadsFromPool);
				PLNetDyadRanker plNet = new PLNetDyadRanker();
				ActiveDyadRanker activeRanker = null;
				if (samplingStrategy.equals("prototypical")) {
					activeRanker = new PrototypicalPoolBasedActiveDyadRanker(plNet, poolProvider, minibatchSize,
							lengthOfTopRankingToConsider, ratioOfOldInstancesInMinibatch);
				} else if (samplingStrategy.equals("random")) {
					activeRanker = new RandomPoolBasedActiveDyadRanker(plNet, poolProvider, minibatchSize, seed);
				} else {
					throw new IllegalArgumentException("Please choose a valid sampling strategy!");
				}

				double currentScore = 0;
				
				for (int iteration = 0; iteration < numberQueries; iteration++) {
					activeRanker.activelyTrain(1);
					currentScore = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), testData, plNet);
				
					Map<String, String> valueMap = new HashMap<>();
					valueMap.put("seed", Integer.toString(seed));
					valueMap.put("query_step", Integer.toString(iteration));
					valueMap.put("sampling_strategy", samplingStrategy);
					valueMap.put("train_ratio", Double.toString(trainRatio));
					valueMap.put("old_data_in_minibatch_ratio", Double.toString(ratioOfOldInstancesInMinibatch));
					valueMap.put("minibatch_size", Integer.toString(minibatchSize));
					valueMap.put("score", Double.toString(currentScore));
					valueMap.put("evaluation_date",
							new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(Date.from(Instant.now())));
					adapter.insert(curveTable, valueMap);
				}

				/* run experiment */
				Map<String, Object> results = new HashMap<>();

				/* report results */
				results.put("measure", "Kendalls Tau");
				results.put("score", currentScore);
				processor.processResults(results);
			}
		});
		runner.randomlyConductExperiments(true);
	}

}
