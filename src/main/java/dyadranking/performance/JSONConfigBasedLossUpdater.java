package dyadranking.performance;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.json.simple.JSONObject;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;
import jaicore.ml.dyadranking.loss.KendallsTauOfTopK;
import jaicore.ml.dyadranking.util.DyadMinMaxScaler;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class JSONConfigBasedLossUpdater {

	private static String pathToConfigFolder = "/Users/elppa/git_pg/DRACO/experiments/jsonConfigs";

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples_full";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final String metaFeatureName = "X_LANDMARKERS";

	private static final int rankingLength = 20;

	private static Random random = new Random(42);

	private static int[] kLengths = { 3, 5, 10 };

	private static Pattern arrayDeserializer = Pattern.compile(" ");
	
	List<Double>[] lossesForMarcelsAvgScore =new List[] {new ArrayList<>(), new ArrayList<>(), new ArrayList<>()};
	
	

	private static Map<String, List<DatabaseDyad>> cachedDyads = new HashMap<>();

	public static void main(String[] args) throws Exception {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		if (args.length > 3 && pathToConfigFolder == null) {
			pathToConfigFolder = args[4];
		}

		File jsonFolder = new File(pathToConfigFolder);
		File[] configs = jsonFolder.listFiles();
		configs = Arrays.stream(configs).filter(File::isFile).filter(f -> f.getName().contains(".json"))
				.toArray(File[]::new);

		Arrays.sort(configs, (f1, f2) -> Integer.compare(Integer.parseInt(f1.getName().replaceAll(".json", "")),
				Integer.parseInt(f2.getName().replaceAll(".json", ""))));

		for (File potentiallyConfig : configs) {

			boolean isAConfigFile = !potentiallyConfig.isDirectory() && potentiallyConfig.getName().contains(".json");
			if (!isAConfigFile) {
				continue;
			}

			JSONObject jsonObject = new ObjectMapper().readValue(potentiallyConfig, JSONObject.class);
			if (!jsonObject.containsKey("rankerWithSer") || !jsonObject.containsKey("ranker")
					|| !new File((String) jsonObject.get("rankerWithSer")).exists()
					|| !new File((String) jsonObject.get("ranker")).exists()) {
				System.out.println("Not all rankers found for " + potentiallyConfig.getName());
				continue;
			}

			String rankerPath = (String) jsonObject.get("ranker");
			String normalizedRankerPath = (String) jsonObject.get("rankerWithSer");

			String scalerPath = (String) jsonObject.get("normalizeSerPath");
			@SuppressWarnings("unchecked")
			List<Integer> testDatasets = (List<Integer>) jsonObject.get("testDatasets");

			@SuppressWarnings("unchecked")
			List<Integer> trainDatasets = (List<Integer>) jsonObject.get("trainDatasets");

			cacheTrainDyads(trainDatasets,adapter, potentiallyConfig.getName());

			// System.out.println("testingDatasets are ..." + testDatasets.toString());
			PLNetDyadRanker ranker = new PLNetDyadRanker();

			ranker.loadModelFromFile(rankerPath);

			PLNetDyadRanker normalizedRanker = new PLNetDyadRanker();
			normalizedRanker.loadModelFromFile(normalizedRankerPath);

			// Alexanders baseline loss init
			List<Instance> datasetsAsInstance = new ArrayList<>();
			int classIndex = 0;
			ArrayList<Attribute> attributes = null;
			for (int trainDatasetId : trainDatasets) {
				double[] landmarkers = getLandmarkersForDatasetId(trainDatasetId, adapter);
				Instance landmarkerInstance = new DenseInstance(landmarkers.length + 1);
				for (int i = 0; i < landmarkers.length; i++) {
					landmarkerInstance.setValue(i, landmarkers[i]);
				}
				if (attributes == null) {
					attributes = new ArrayList<>();
					for (int i = 0; i < landmarkers.length; i++) {
						attributes.add(new Attribute("landmarker_" + i));
					}
					attributes.add(new Attribute("datasetId"));
					classIndex = landmarkers.length;
				}
				// use the id as label such that ibk can directly return it
				landmarkerInstance.setValue(landmarkers.length, trainDatasetId);
				datasetsAsInstance.add(landmarkerInstance);
			}

			Instances trainDatasetIdInstances = new Instances("DatasetInstances", attributes,
					datasetsAsInstance.size());
			trainDatasetIdInstances.addAll(datasetsAsInstance);
			trainDatasetIdInstances.setClassIndex(classIndex);

			// 1NN
			IBk ibk = new IBk(1);
			ibk.buildClassifier(trainDatasetIdInstances);

			try (ObjectInputStream oin = new ObjectInputStream(new FileInputStream(scalerPath))) {
				DyadMinMaxScaler scaler = (DyadMinMaxScaler) oin.readObject();

				double avgKT = 0.0d;
				double avgKT_withNorm = 0.0d;

				double avgKT_Baseline_1NN = 0.0d;
				double avgKT_Baseline_AverageRank = 0.0d;
				double[] avgTopKTauDistance = { 0.0d, 0.0d, 0.0d };
				double[] avgTopKTauDistance_baseline_1NN = { 0.0d, 0.0d, 0.0d };
				double[] avgTopKTauDistance_baseline_AverageRank = { 0.0d, 0.0d, 0.0d };
				double[] avgMarcelsLossWithMin = { 0.0d, 0.0d, 0.0d };
				double[] avgMarcelsLossWithAvg = { 0.0d, 0.0d, 0.0d };
				double[] avgMarcelsLossWithAvgToBaseline_1NN = { 0.0d, 0.0d, 0.0d };
				double[] avgMarcelsLossWithMinToBaseline_1NN = { 0.0d, 0.0d, 0.0d };
				double[] avgMarcelsLossWithAvgToBaseline_AverageRank = { 0.0d, 0.0d, 0.0d };
				double[] avgMarcelsLossWithMinToBaseline_AverageRank = { 0.0d, 0.0d, 0.0d };

				for (int datasetId : testDatasets) {

					// 1 - NN Setup:
					// Find the nearest dataset in the training data according to the dataset
					// meatfeatures
					double[] metaFeatres = getLandmarkersForDatasetId(datasetId, adapter);
					Instance instance = new DenseInstance(metaFeatres.length + 1);
					instance.setDataset(trainDatasetIdInstances);
					for (int i = 0; i < metaFeatres.length; i++) {
						instance.setValue(i, metaFeatres[i]);
					}
					instance.setClassMissing();
					// the label is just the dataset id -> only works for 1 NN !!!
					double closestDatasetId = ibk.classifyInstance(instance);

					/* Averaging setup -> For every Dataset we draw 10 DyadRankings */

					/* Losses for our approach */
					double innerAVG = 0.0d;
					double innerAVGWithNorm = 0.0d;

					double[] innerAVGTopKKTDistance = { 0.0d, 0.0d, 0.0d };

					double[] innerMarcelsLossWithMin = { 0.0d, 0.0d, 0.0d };

					double[] innerMarcelsLossWithAvg = { 0.0d, 0.0d, 0.0d };

					/* 1 NN Baseline losses */

					double innerAverageKT_Baseline_1NN = 0.0d;

					double[] innerMarcelsLossWithMinBaseline_1NN = { 0.0d, 0.0d, 0.0d };

					double[] innerMarcelsLossWithAvgBaseline_1NN = { 0.0d, 0.0d, 0.0d };

					double[] innerAverageTopKKT_Baseline_1NN = { 0.0d, 0.0d, 0.0d };

					/* Average-Rank Baseline losses */
					double innerAverageKT_Baseline_AverageRank = 0.0d;

					double[] innerAverageTopKKT_Baseline_AverageRank = { 0.0d, 0.0d, 0.0d };

					double[] innerMarcelsLossWithMinBaseline_AverageRank = { 0.0d, 0.0d, 0.0d };

					double[] innerMarcelsLossWithAvgBaseline_AverageRank = { 0.0d, 0.0d, 0.0d };

					/* Now do 10 repeats for averaging */
					for (int i = 0; i < 10; i++) {

						/*
						 * Draw the dyads that should be ranked (careful that the dyads have a distinct
						 * score to not cunfuse the Kendalls Taus measures)
						 */
						List<Pair<Double, Dyad>> testDyads = getDyadRankingInstanceForDataset(datasetId, adapter);

						List<Dyad> orderedDyads = testDyads.stream().sorted(DyadComparator::compare).map(Pair::getY)
								.collect(Collectors.toList());
						DyadRankingDataset trueOrdering = DyadRankingDataset.fromOrderedDyadList(orderedDyads);

						/* Now get the prediction for the 1 NN classifier */
						// just be careful that the ordering of the dyads do not affect the baseline
						// here
						Collections.shuffle(testDyads, random);
						// returns an already ordered list!
						List<Pair<Double, Dyad>> baseline_1NN_dyadPairs = get1NNRanking(closestDatasetId, adapter,
								testDyads.stream().map(Pair::getY).collect(Collectors.toList()),
								potentiallyConfig.getName());

						List<Dyad> baseline_1NN_orderedDyads = baseline_1NN_dyadPairs.stream().map(Pair::getY)
								.collect(Collectors.toList());
						DyadRankingDataset baseline_1NN_predictedRanking = DyadRankingDataset
								.fromOrderedDyadList(baseline_1NN_orderedDyads);

						/* Now get the predicition of the AverageRank baseline */
						// again, be careful that no ordering information influences this
						Collections.shuffle(testDyads, random);
						// returns an already ordered list!
						List<Pair<Double, Dyad>> baseline_AverageRank_dyadPairs = getAverageRankForDatasets(adapter,
								testDyads.stream().map(Pair::getY).collect(Collectors.toList()),
								potentiallyConfig.getName());
						List<Dyad> baseline_AverageRank_orderedDyads = baseline_AverageRank_dyadPairs.stream()
								.map(Pair::getY).collect(Collectors.toList());
						DyadRankingDataset baseline_AverageRank_predictedRanking = DyadRankingDataset
								.fromOrderedDyadList(baseline_AverageRank_orderedDyads);

						/* Used for marcels losses on the baselines */
						Map<Object, Object> map = testDyads.stream().collect(Collectors.toMap(Pair::getY, Pair::getX));

						for (int j = 0; j < kLengths.length; j++) {
							innerAVGTopKKTDistance[j] += DyadRankingLossUtil.computeAverageLoss(
									new KendallsTauOfTopK(kLengths[j], 0.5d), trueOrdering, ranker, random);
						}

						// re-order the true ordering, in case that the loss function messes this up!
						orderedDyads = testDyads.stream().sorted(DyadComparator::compare).map(Pair::getY)
								.collect(Collectors.toList());
						trueOrdering = DyadRankingDataset.fromOrderedDyadList(orderedDyads);

						// 1NN Baseline KT
						innerAverageKT_Baseline_1NN += DyadRankingLossUtil.computeAverageLoss(
								new KendallsTauDyadRankingLoss(), trueOrdering, baseline_1NN_predictedRanking);

						// again reordering time!
						orderedDyads = testDyads.stream().sorted(DyadComparator::compare).map(Pair::getY)
								.collect(Collectors.toList());
						trueOrdering = DyadRankingDataset.fromOrderedDyadList(orderedDyads);

						// Average Rank KT
						innerAverageKT_Baseline_AverageRank += DyadRankingLossUtil.computeAverageLoss(
								new KendallsTauDyadRankingLoss(), trueOrdering, baseline_AverageRank_predictedRanking);

						List<Dyad> shuffelledDyads = testDyads.stream().map(Pair::getY).collect(Collectors.toList());
						Collections.shuffle(shuffelledDyads, random);
						// can be used for the predictions of our approach
						DyadRankingDataset shuffeledOrdering = DyadRankingDataset.fromOrderedDyadList(orderedDyads);

						// Now calculate Marcels losses:
						for (int j = 0; j < kLengths.length; j++) {
							int k = kLengths[j];

							// Top K-Kendalls distance losses
							innerAverageTopKKT_Baseline_1NN[j] += DyadRankingLossUtil.computeAverageLoss(
									new KendallsTauOfTopK(k, 0.5d), trueOrdering, baseline_1NN_predictedRanking);

							innerAverageTopKKT_Baseline_AverageRank[j] += DyadRankingLossUtil.computeAverageLoss(
									new KendallsTauOfTopK(k, 0.5d), trueOrdering,
									baseline_AverageRank_predictedRanking);

							List<IDyadRankingInstance> predicted = ranker.predict(shuffeledOrdering);
							List<Dyad> topKPredictedDyads = new ArrayList<>();
							for (int l = 0; l < k; l++) {
								topKPredictedDyads.add(predicted.get(0).getDyadAtPosition(l));
							}
							// marcels loss to predicted ranking
							double predictedMin = topKPredictedDyads.stream().mapToDouble(x -> (Double) map.get(x))
									.min().orElseThrow(NoSuchElementException::new);
							double realMin = testDyads.stream().sorted(DyadComparator::compare)
									.collect(Collectors.toList()).subList(0, k).stream().mapToDouble(Pair::getX).min()
									.orElseThrow(NoSuchElementException::new);
							double marcelsMinLoss = Math.abs(predictedMin - realMin);
							double predictedAvg = topKPredictedDyads.stream().mapToDouble(x -> (Double) map.get(x))
									.average().orElseThrow(NoSuchElementException::new);
							double realAvg = testDyads.stream().sorted(DyadComparator::compare)
									.collect(Collectors.toList()).subList(0, k).stream().mapToDouble(Pair::getX)
									.average().orElseThrow(NoSuchElementException::new);
							double marcelsAvgLoss = Math.abs(predictedAvg - realAvg);
							innerMarcelsLossWithAvg[j] += marcelsAvgLoss;
							innerMarcelsLossWithMin[j] += marcelsMinLoss;

							// marcels loss for baseline ranking 1NN
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
							innerMarcelsLossWithAvgBaseline_1NN[j] += baseline_1NN_AvgLoss;
							innerMarcelsLossWithMinBaseline_1NN[j] += baseline_1NN_MinLoss;

							// marcels loss for baseline average rank
							List<Pair<Double, Dyad>> topKOfBaseLine_AverageRank = baseline_AverageRank_dyadPairs
									.subList(0, k);
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
							innerMarcelsLossWithAvgBaseline_AverageRank[j] += baseline_AverageRank_AvgLoss;
							innerMarcelsLossWithMinBaseline_AverageRank[j] += baseline_AverageRank_MinLoss;

						}

						innerAVG += DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(),
								trueOrdering, ranker, random);

						scaler.transformAlternatives(shuffeledOrdering);
						innerAVGWithNorm += DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(),
								trueOrdering, normalizedRanker, random);

					}
					for (int i = 0; i < kLengths.length; i++) {
						avgTopKTauDistance_baseline_1NN[i] += innerAverageTopKKT_Baseline_1NN[i] / 10.0d;
						avgTopKTauDistance_baseline_AverageRank[i] += innerAverageTopKKT_Baseline_AverageRank[i]
								/ 10.0d;
						avgTopKTauDistance[i] += innerAVGTopKKTDistance[i] / 10.0d;
						avgMarcelsLossWithAvg[i] += innerMarcelsLossWithAvg[i] / 10.0d;
						avgMarcelsLossWithMin[i] += innerMarcelsLossWithMin[i] / 10.0d;
						avgMarcelsLossWithAvgToBaseline_1NN[i] += innerMarcelsLossWithAvgBaseline_1NN[i] / 10.0d;
						avgMarcelsLossWithMinToBaseline_1NN[i] += innerMarcelsLossWithMinBaseline_1NN[i] / 10.0d;
						avgMarcelsLossWithAvgToBaseline_AverageRank[i] += innerMarcelsLossWithAvgBaseline_AverageRank[i]
								/ 10.0d;
						avgMarcelsLossWithMinToBaseline_AverageRank[i] += innerMarcelsLossWithMinBaseline_AverageRank[i]
								/ 10.0d;

					}
					avgKT_Baseline_1NN += innerAverageKT_Baseline_1NN / 10.0d;
					avgKT_Baseline_AverageRank += innerAverageKT_Baseline_AverageRank / 10.0d;

					avgKT += innerAVG / 10.0d;
					avgKT_withNorm += innerAVGWithNorm / 10.0d;

				}
				avgKT_Baseline_1NN = avgKT_Baseline_1NN / testDatasets.size();
				avgKT_Baseline_AverageRank = avgKT_Baseline_1NN / testDatasets.size();

				avgKT = avgKT / testDatasets.size();
				avgKT_withNorm = avgKT_withNorm / testDatasets.size();

				// print the results
				System.out.println("---" + potentiallyConfig.getName() + "---");
				for (int i = 0; i < kLengths.length; i++) {
					System.out.println("K = " + kLengths[i]);
					avgTopKTauDistance_baseline_1NN[i] = avgTopKTauDistance_baseline_1NN[i] / testDatasets.size();
					avgTopKTauDistance_baseline_AverageRank[i] = avgTopKTauDistance_baseline_AverageRank[i]
							/ testDatasets.size();

					avgTopKTauDistance[i] = avgTopKTauDistance[i] / testDatasets.size();
					avgMarcelsLossWithAvg[i] = avgMarcelsLossWithAvg[i] / testDatasets.size();
					avgMarcelsLossWithMin[i] = avgMarcelsLossWithMin[i] / testDatasets.size();
					avgMarcelsLossWithAvgToBaseline_1NN[i] += avgMarcelsLossWithAvgToBaseline_1NN[i]
							/ testDatasets.size();
					avgMarcelsLossWithMinToBaseline_1NN[i] += avgMarcelsLossWithMinToBaseline_1NN[i]
							/ testDatasets.size();
					avgMarcelsLossWithAvgToBaseline_AverageRank[i] += avgMarcelsLossWithAvgToBaseline_AverageRank[i]
							/ testDatasets.size();
					avgMarcelsLossWithMinToBaseline_AverageRank[i] += avgMarcelsLossWithMinToBaseline_AverageRank[i]
							/ testDatasets.size();
					System.out.println("Average Marcels Loss for Avg score from the baseline that uses 1NN is  "
							+ avgMarcelsLossWithAvgToBaseline_1NN[i]);
					System.out.println("Average Marcels Loss for Min score from the baseline that uses 1NN is  "
							+ avgMarcelsLossWithMinToBaseline_1NN[i]);

					System.out
							.println("Average Marcels Loss for Avg score from the baseline that uses Average-Rank is  "
									+ avgMarcelsLossWithAvgToBaseline_AverageRank[i]);
					System.out
							.println("Average Marcels Loss for Min score from the baseline that uses Average-Rank is  "
									+ avgMarcelsLossWithMinToBaseline_AverageRank[i]);

					System.out.println(
							"Average Marcels Loss for Avg score (our approach) is " + avgMarcelsLossWithAvg[i]);
					System.out.println(
							"Average Marcels Loss for Min score (our approach) is " + avgMarcelsLossWithMin[i]);
					System.out.println("Average Top K KTau Distance " + avgTopKTauDistance[i]);
					System.out.println(
							"Average Top K KTau Distance for baseline 1NN " + avgTopKTauDistance_baseline_1NN[i]);
					System.out.println("Average Top K KTau Distance for baseline AverageRank "
							+ avgTopKTauDistance_baseline_AverageRank[i]);

					System.out.println("--------------");
				}

				System.out.println("Kendalls Tau without norm " + avgKT);
				System.out.println("Kendalls Tau for baseline 1NN " + avgKT_Baseline_1NN);
				System.out.println("Kendalls Tau for baseline AverageRank " + avgKT_Baseline_AverageRank);
				// System.out.println("Kendalls Tau with norm " + avgKT_withNorm);
			} catch (IOException | ClassNotFoundException e) {
				System.err.println("Failed to deserialize scaler!");
				return;
			}

		}
	}

	private static void cacheTrainDyads(List<Integer> trainDatasets, SQLAdapter adapter, String key)
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

	private static List<Pair<Double, Dyad>> get1NNRanking(double closestDatasetId, SQLAdapter adapter, List<Dyad> list,
			String key) throws Exception {

		List<Pair<Double, Dyad>> toReturn = new ArrayList<>();
		List<DatabaseDyad> dyadsInDB = cachedDyads.get(key);
		for (Dyad dyad : list) {
			String serializedY = dyad.getAlternative().stream().boxed().map(d -> d.toString())
					.collect(Collectors.joining(" "));
			Optional<DatabaseDyad> matchingDyad = dyadsInDB.stream()
					.filter(d -> d.getDatasetId() == closestDatasetId && d.getSerializedY().equals(serializedY))
					.findFirst();
			if (matchingDyad.isPresent()) {
				// there are some pipelines missing
				toReturn.add(new Pair<>(matchingDyad.get().getScore(), dyad));
			} else {
				// we need all
				toReturn.add(new Pair<>(1.0, dyad));
			}
		}
		Collections.sort(toReturn, DyadComparator::compare);
		return toReturn;
	}

	/**
	 * Average rank approach: For each dyad it takes the average performance score
	 * form the train datasets and ranks the dyad according to this score.
	 * 
	 * @param datasets
	 *            the train datasets
	 * @param adapter
	 * @param toRank
	 *            the dyads to rank
	 * @return
	 * @throws SQLException
	 */
	private static List<Pair<Double, Dyad>> getAverageRankForDatasets(SQLAdapter adapter, List<Dyad> toRank, String key)
			throws SQLException {
		List<Pair<Double, Dyad>> toReturn = new ArrayList<>();
		List<DatabaseDyad> cache = cachedDyads.get(key);
		for (Dyad dyad : toRank) {
			String serializedY = dyad.getAlternative().stream().boxed().map(d -> d.toString())
					.collect(Collectors.joining(" "));
			OptionalDouble optScore = cache.stream().filter(d -> d.getSerializedY().equals(serializedY))
					.mapToDouble(DatabaseDyad::getScore).average();
			if (optScore.isPresent()) {
				// there are some pipelines missing
				toReturn.add(new Pair<>(optScore.getAsDouble(), dyad));
			} else {
				toReturn.add(new Pair<>(1.0, dyad));
			}
		}
		Collections.sort(toReturn, DyadComparator::compare);
		return toReturn;
	}

	private static List<Pair<Double, Dyad>> getDyadRankingInstanceForDataset(int datasetId, SQLAdapter adapter)
			throws Exception {
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

	private static double[] getLandmarkersForDatasetId(int datasetId, SQLAdapter adapter) throws SQLException {
		ResultSet rs = adapter.getResultsOfQuery(
				"SELECT " + metaFeatureName + " FROM " + datasetMetaFeatureTable + " WHERE dataset = " + datasetId);

		rs.first();
		String serX = rs.getString(1);
		return arrayDeserializer.splitAsStream(serX).mapToDouble(Double::parseDouble).toArray();
	}
}
