package dyadranking.performance.mccvevaluation.experimenters;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Time;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Properties;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;
import org.json.simple.JSONObject;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import dyadranking.performance.mccvevaluation.APPROACH;
import dyadranking.performance.mccvevaluation.AverageRankBaseline;
import dyadranking.performance.mccvevaluation.DatabaseDyad;
import dyadranking.performance.mccvevaluation.DyadComparator;
import dyadranking.performance.mccvevaluation.JSONConfigKeys;
import dyadranking.performance.mccvevaluation.LossUpdater;
import dyadranking.performance.mccvevaluation.OneNNBaseline;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;
import jaicore.ml.dyadranking.loss.KendallsTauOfTopK;

public class MarcelsLossExperimenter implements IExperimentSetEvaluator {

	private static String pathToConfigFolder = null;

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples_with_SMO";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final String metaFeatureName = "X_LANDMARKERS";

	private static final int rankingLength = 20;

	private static int[] kLengths = { 3, 5, 10 };

	private static Pattern arrayDeserializer = Pattern.compile(" ");

	static int submittedTasks = 0;

	static int avgRun;

	private static Map<String, List<DatabaseDyad>> cachedDyads = new HashMap<>();

	private MarcelsLossExperimenterConfig config;

	public MarcelsLossExperimenter(File configFile) {
		Properties props = new Properties();
		try {
			props.load(new FileInputStream(configFile));
		} catch (IOException e) {
			System.err.println("Could not find or access config file " + configFile);
			System.exit(1);
		}
		this.config = ConfigFactory.create(MarcelsLossExperimenterConfig.class, props);
	}

	@Override
	public IExperimentSetConfig getConfig() {
		return config;
	}

	@Override
	public void evaluate(ExperimentDBEntry experimentEntry, SQLAdapter adapter,
			IExperimentIntermediateResultProcessor processor) throws Exception {
		// Put ArrayjobNr in fields to identify error logs from db table
		Map<String, Object> results = new HashMap<>();

		Map<String, String> keyfields = experimentEntry.getExperiment().getValuesOfKeyFields();
		System.out.println("Start experiment with keys: " + keyfields);
		LossUpdater lossUpdater = new LossUpdater(adapter, "approach_5_evals_final_" + avgRun);

		File potentiallyConfig = new File(pathToConfigFolder + File.separatorChar + keyfields.get("config"));
		try {
			if (!isAValidConfig(potentiallyConfig)) {
				return;
			}

			JSONObject jsonObject = new ObjectMapper().readValue(potentiallyConfig, JSONObject.class);

			System.out.println("Starting " + potentiallyConfig.getName());
			// start evaluating here

			// outer loop: mccv split

			String rankerPath = (String) jsonObject.get(JSONConfigKeys.RANKER_PATH);

			// String normalizedRankerPath = (String)
			// jsonObject.get(JSONConfigKeys.RANKER_WITH_NORMALIZATION);

			// String scalerPath = (String) jsonObject.get(JSONConfigKeys.SCALER_PATH);

			@SuppressWarnings("unchecked")
			List<Integer> testDatasets = (List<Integer>) jsonObject.get(JSONConfigKeys.TEST_DATASETS);

			@SuppressWarnings("unchecked")
			List<Integer> trainDatasets = (List<Integer>) jsonObject.get(JSONConfigKeys.TRAIN_DATASETS);

			int mccvSplit = (int) jsonObject.get(JSONConfigKeys.MCCV_INDEX);

			int subSamplingSize = (int) jsonObject.get(JSONConfigKeys.SUBSAMPLING_SIZE);

			String cacheKey = Integer.toString(mccvSplit);
			System.out.println("Start caching dyads for speedup...");
			cacheTrainDyads(trainDatasets, adapter, cacheKey);

			// System.out.println("testingDatasets are ..." + testDatasets.toString());
			PLNetDyadRanker ranker = new PLNetDyadRanker();

			ranker.loadModelFromFile(rankerPath);

			// PLNetDyadRanker normalizedRanker = new PLNetDyadRanker();
			// normalizedRanker.loadModelFromFile(normalizedRankerPath);

			System.out.println("Start to train 1NN Baseline...");
			OneNNBaseline oneNNBaseline = new OneNNBaseline(cachedDyads);
			oneNNBaseline.buildClassifier(trainDatasets, adapter);

			AverageRankBaseline averageRankBaseline = new AverageRankBaseline(cachedDyads);

			// try (ObjectInputStream oin = new ObjectInputStream(new
			// FileInputStream(scalerPath))) {

			// DyadMinMaxScaler scaler = (DyadMinMaxScaler) oin.readObject();

			for (int testDatasetId : testDatasets) {
				System.out.println("Starting testdataset " + testDatasetId + " for " + potentiallyConfig.getName());

				double closestDatasetId = oneNNBaseline.getClosestDatasetId(testDatasetId, adapter);

				/* Now do 10 repeats for averaging */
				for (int innerAveragingRun = 0; innerAveragingRun < avgRun; innerAveragingRun++) {

					Random random = new Random(innerAveragingRun + mccvSplit);

					/*
					 * Draw the dyads that should be ranked (careful that the dyads have a distinct
					 * score to not cunfuse the Kendalls Taus measures)
					 */
					List<Pair<Double, Dyad>> testDyads = getDyadRankingInstanceForDataset(testDatasetId, adapter,
							random);

					List<Dyad> orderedDyads = testDyads.stream().sorted(DyadComparator::compare).map(Pair::getY)
							.collect(Collectors.toList());
					DyadRankingDataset trueOrdering = DyadRankingDataset.fromOrderedDyadList(orderedDyads);

					/* Now get the prediction for the 1 NN classifier */
					// just be careful that the ordering of the dyads do not affect the baseline
					// here
					Collections.shuffle(testDyads, random);
					List<Dyad> shuffeledDyads = testDyads.stream().map(Pair::getY).collect(Collectors.toList());

					List<Pair<Double, Dyad>> baseline_1NN_dyadPairs = oneNNBaseline.get1NNRanking(closestDatasetId,
							shuffeledDyads, cacheKey);
					List<Dyad> baseline_1NN_orderedDyads = baseline_1NN_dyadPairs.stream().map(Pair::getY)
							.collect(Collectors.toList());
					DyadRankingDataset baseline_1NN_predictedRanking = DyadRankingDataset
							.fromOrderedDyadList(baseline_1NN_orderedDyads);

					/* Now get the predicition of the AverageRank baseline */
					// again, be careful that no ordering information influences this

					Collections.shuffle(testDyads, random);
					shuffeledDyads = testDyads.stream().map(Pair::getY).collect(Collectors.toList());
					// returns an already ordered list!
					List<Pair<Double, Dyad>> baseline_AverageRank_dyadPairs = averageRankBaseline
							.getAverageRankForDatasets(shuffeledDyads, cacheKey);

					List<Dyad> baseline_AverageRank_orderedDyads = baseline_AverageRank_dyadPairs.stream()
							.map(Pair::getY).collect(Collectors.toList());
					DyadRankingDataset baseline_AverageRank_predictedRanking = DyadRankingDataset
							.fromOrderedDyadList(baseline_AverageRank_orderedDyads);

					/* Used for marcels losses on the baselines */
					Map<Object, Object> map = testDyads.stream().collect(Collectors.toMap(Pair::getY, Pair::getX));

					for (int j = 0; j < kLengths.length; j++) {
						double topKKTAU = DyadRankingLossUtil.computeAverageLoss(
								new KendallsTauOfTopK(kLengths[j], 0.5d), trueOrdering, ranker, random);
						lossUpdater.updateTopKKTAULoss(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
								kLengths[j], APPROACH.DYADRANKING, topKKTAU);
					}

					// re-order the true ordering, in case that the loss function messes this up!
					orderedDyads = testDyads.stream().sorted(DyadComparator::compare).map(Pair::getY)
							.collect(Collectors.toList());
					trueOrdering = DyadRankingDataset.fromOrderedDyadList(orderedDyads);

					// 1NN Baseline KT

					double kTau_1NN = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(),
							trueOrdering, baseline_1NN_predictedRanking);

					lossUpdater.updateKendallsTau(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
							APPROACH.ONE_NN_BASELINE, kTau_1NN);

					// again reordering time!
					orderedDyads = testDyads.stream().sorted(DyadComparator::compare).map(Pair::getY)
							.collect(Collectors.toList());
					trueOrdering = DyadRankingDataset.fromOrderedDyadList(orderedDyads);

					// Average Rank KT
					double kTau_AverageRank = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(),
							trueOrdering, baseline_AverageRank_predictedRanking);

					lossUpdater.updateKendallsTau(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
							APPROACH.AVERAGE_RANK_BASELINE, kTau_AverageRank);

					List<Dyad> shuffelledDyads = testDyads.stream().map(Pair::getY).collect(Collectors.toList());
					Collections.shuffle(shuffelledDyads, random);
					// can be used for the predictions of our approach
					DyadRankingDataset shuffeledOrdering = DyadRankingDataset.fromOrderedDyadList(orderedDyads);

					// Now calculate Marcels losses:
					for (int j = 0; j < kLengths.length; j++) {
						int k = kLengths[j];

						// Top K-Kendalls distance losses
						double topKKTAU_1NN = DyadRankingLossUtil.computeAverageLoss(new KendallsTauOfTopK(k, 0.5d),
								trueOrdering, baseline_1NN_predictedRanking);
						lossUpdater.updateTopKKTAULoss(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun, k,
								APPROACH.ONE_NN_BASELINE, topKKTAU_1NN);

						double topKKTAU_AverageRank = DyadRankingLossUtil.computeAverageLoss(
								new KendallsTauOfTopK(k, 0.5d), trueOrdering, baseline_AverageRank_predictedRanking);
						lossUpdater.updateTopKKTAULoss(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun, k,
								APPROACH.AVERAGE_RANK_BASELINE, topKKTAU_AverageRank);

						List<IDyadRankingInstance> predicted = ranker.predict(shuffeledOrdering);
						List<Dyad> topKPredictedDyads = new ArrayList<>();
						for (int l = 0; l < k; l++) {
							topKPredictedDyads.add(predicted.get(0).getDyadAtPosition(l));
						}

						// wever number min
						double predictedMin = topKPredictedDyads.stream().mapToDouble(x -> (Double) map.get(x)).min()
								.orElseThrow(NoSuchElementException::new);
						double realMin = testDyads.stream().sorted(DyadComparator::compare).limit(k)
								.mapToDouble(Pair::getX).min().orElseThrow(NoSuchElementException::new);
						double weverNumberMin = Math.abs(predictedMin - realMin);

						// wever number avg
						double predictedAvg = topKPredictedDyads.stream().mapToDouble(x -> (Double) map.get(x))
								.average().orElseThrow(NoSuchElementException::new);
						double realAvg = testDyads.stream().sorted(DyadComparator::compare).limit(k)
								.mapToDouble(Pair::getX).average().orElseThrow(NoSuchElementException::new);
						double weverNumberAvg = Math.abs(predictedAvg - realAvg);

						lossUpdater.updateWeverNumberAvg(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
								k, APPROACH.DYADRANKING, weverNumberAvg);
						lossUpdater.updateWeverNumberMin(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
								k, APPROACH.DYADRANKING, weverNumberMin);

						// wevre number for baseline ranking 1NN
						List<Pair<Double, Dyad>> topKOfBaseLine_1NN = baseline_1NN_dyadPairs.subList(0, k);
						// attention here: the scores are the scores from the wrong dataset, we have to
						// map them back first!
						double avgOfBaseline_1NN = topKOfBaseLine_1NN.stream()
								.mapToDouble(x -> (Double) map.get(x.getY())).average()
								.orElseThrow(NoSuchElementException::new);
						double minOfBaseline_1NN = topKOfBaseLine_1NN.stream()
								.mapToDouble(x -> (Double) map.get(x.getY())).min()
								.orElseThrow(NoSuchElementException::new);

						double baseline_1NN_AvgLoss = Math.abs(avgOfBaseline_1NN - realAvg);
						double baseline_1NN_MinLoss = Math.abs(minOfBaseline_1NN - realMin);

						lossUpdater.updateWeverNumberAvg(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
								k, APPROACH.ONE_NN_BASELINE, baseline_1NN_AvgLoss);
						lossUpdater.updateWeverNumberMin(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
								k, APPROACH.ONE_NN_BASELINE, baseline_1NN_MinLoss);

						// wever number for baseline average rank
						List<Pair<Double, Dyad>> topKOfBaseLine_AverageRank = baseline_AverageRank_dyadPairs.subList(0,
								k);
						// attention here: the scores are the scores from the wrong dataset, we have to
						// map them back first!
						double avgOfBaseline_AverageRank = topKOfBaseLine_AverageRank.stream()
								.mapToDouble(x -> (Double) map.get(x.getY())).average()
								.orElseThrow(NoSuchElementException::new);
						double minOfBaseline_AverageRank = topKOfBaseLine_AverageRank.stream()
								.mapToDouble(x -> (Double) map.get(x.getY())).min()
								.orElseThrow(NoSuchElementException::new);

						double baseline_AverageRank_AvgLoss = Math.abs(avgOfBaseline_AverageRank - realAvg);
						double baseline_AverageRank_MinLoss = Math.abs(minOfBaseline_AverageRank - realMin);

						lossUpdater.updateWeverNumberAvg(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
								k, APPROACH.AVERAGE_RANK_BASELINE, baseline_AverageRank_AvgLoss);
						lossUpdater.updateWeverNumberMin(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
								k, APPROACH.AVERAGE_RANK_BASELINE, baseline_AverageRank_MinLoss);

					}

					double kendallsTau = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(),
							trueOrdering, ranker, random);

					lossUpdater.updateKendallsTau(subSamplingSize, mccvSplit, testDatasetId, innerAveragingRun,
							APPROACH.DYADRANKING, kendallsTau);

					// scaler.transformAlternatives(shuffeledOrdering);
					// double normalizedKendallsTau = DyadRankingLossUtil.computeAverageLoss(
					// new KendallsTauDyadRankingLoss(), trueOrdering, normalizedRanker, random);
					// lossUpdater.updateKendallsTauAverage(subSamplingSize, mccvSplit,
					// testDatasetId,
					// innerAveragingRun, APPROACH.DYADRANKING, normalizedKendallsTau);

				}
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		results.put("done", "1");
		processor.processResults(results);
	}

	private static void print(final String message) {
		System.out.println(new Time(System.currentTimeMillis()).toString() + ": " + message);
	}

	public static void main(final String[] args) {
		if (args.length > 3 && pathToConfigFolder == null) {
			pathToConfigFolder = args[4];
			avgRun = Integer.parseInt(args[5]);
		}

		/* check config */
		print("Start experiment runner...");
		
		print("Conduct random experiment...");
		for (int i = 0; i < 4; i++) {
			new Thread(() -> {
				ExperimentRunner runner = new ExperimentRunner(
						new MarcelsLossExperimenter(new File("conf/approach_5/approach_5_marcels.properties")));
				runner.randomlyConductExperiments(-1, true);
				print("Experiment conducted.");
			}).start();
		}
	}

	private static synchronized void cacheTrainDyads(List<Integer> trainDatasets, SQLAdapter adapter, String key)
			throws SQLException {
		ResultSet rs = adapter.getResultsOfQuery("SELECT score, y, dataset, id FROM " + dyadTable + " WHERE dataset IN "
				+ trainDatasets.stream().map(i -> i.toString()).collect(Collectors.joining(",", "(", ")")));
		List<DatabaseDyad> toReturn = new ArrayList<>();
		while (rs.next()) {
			double score = rs.getDouble(1);
			String serY = rs.getString(2);
			int dataset = rs.getInt(3);
			int id = rs.getInt(4);
			toReturn.add(new DatabaseDyad(dataset, serY, score, id));
		}
		cachedDyads.put(key, toReturn);
	}

	private static List<Pair<Double, Dyad>> getDyadRankingInstanceForDataset(int datasetId, SQLAdapter adapter,
			Random random) throws Exception {
		List<Pair<Double, Dyad>> toReturn = new ArrayList<>();
		ResultSet resSet = adapter.getResultsOfQuery("SELECT id FROM " + dyadTable + " WHERE dataset=" + datasetId);
		List<Integer> allIdsForDataset = new ArrayList<>();
		List<Integer> drawnIndices = new ArrayList<>();
		while (resSet.next()) {
			allIdsForDataset.add(resSet.getInt(1));
		}
		while (toReturn.size() < rankingLength) {
			int randomIndexInList = random.nextInt(allIdsForDataset.size());
			int randomIndex = allIdsForDataset.get(randomIndexInList);
			while (drawnIndices.contains(randomIndex)) {
				randomIndexInList = random.nextInt(allIdsForDataset.size());
				randomIndex = allIdsForDataset.get(randomIndexInList);
			}

			ResultSet resDyads = adapter.getResultsOfQuery("SELECT y, score, " + metaFeatureName + " FROM " + dyadTable
					+ " NATURAL JOIN " + datasetMetaFeatureTable + " WHERE id =" + randomIndex);
			resDyads.first();
			Pair<Double, Dyad> randomDyad = getDyadFromResultSet(resDyads);
			if (toReturn.stream().noneMatch(d -> d.getX().doubleValue() == randomDyad.getX().doubleValue()))
				toReturn.add(randomDyad);
		}
		return toReturn;
	}

	private static Pair<Double, Dyad> getDyadFromResultSet(ResultSet resDyads) throws SQLException {
		String serializedY = resDyads.getString(1);
		Double score = resDyads.getDouble(2);
		String serializedX = resDyads.getString(3);
		double[] xArray = arrayDeserializer.splitAsStream(serializedX).mapToDouble(Double::parseDouble).toArray();
		double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble).toArray();
		return new Pair<>(score, new Dyad(new DenseDoubleVector(xArray), new DenseDoubleVector(yArray)));

	}

	private static boolean isAValidConfig(File potentiallyConfig) throws IOException {
		boolean isAValidConfig = true;
		boolean isAConfigFile = !potentiallyConfig.isDirectory() && potentiallyConfig.getName().contains(".json");
		if (!isAConfigFile) {
			isAValidConfig = false;
		}

		JSONObject jsonObject = new ObjectMapper().readValue(potentiallyConfig, JSONObject.class);
		if (
		// !jsonObject.containsKey(JSONConfigKeys.RANKER_WITH_NORMALIZATION) ||
		!jsonObject.containsKey(JSONConfigKeys.RANKER_PATH)
				// || !new File((String)
				// jsonObject.get(JSONConfigKeys.RANKER_WITH_NORMALIZATION)).exists()
				|| !new File((String) jsonObject.get(JSONConfigKeys.RANKER_PATH)).exists()) {
			System.out.println("Not all rankers found for " + potentiallyConfig.getName());
			isAValidConfig = false;
		}
		return isAValidConfig;
	}

}
