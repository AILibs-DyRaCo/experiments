package dyadranking.performance;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.aeonbits.owner.ConfigFactory;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.dataset.SparseDyadRankingInstance;
import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;
import jaicore.ml.dyadranking.util.AbstractDyadScaler;
import jaicore.ml.dyadranking.util.DyadUnitIntervalScaler;

public class DyadDatasetGenerator {

	private static Integer[] allowedDatasetIds;

	private static String dyadTable;

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final Pattern arrayDeserializer = Pattern.compile(" ");

	private static final String DYAD_FILE = "dyad_pool.txt";

	private static String X_KEY = "X_LANDMARKERS";

	/**
	 * Queries the DB to extract the dyad and its' perfomance score.
	 * 
	 * @param id
	 *            specifies the db entry
	 * @return the dyad
	 * @throws SQLException
	 */
	private static Pair<Dyad, Double> getDyadAndScoreWithId(int id, SQLAdapter adapter) throws SQLException {
		ResultSet res = adapter.getResultsOfQuery("SELECT " + X_KEY + ", score FROM " + dyadTable + " NATURAL JOIN "
				+ datasetMetaFeatureTable + " WHERE id=" + id);
		if (res.wasNull())
			throw new IllegalArgumentException("No entry with id " + id);

		res.first();

		ResultSet res_y = adapter.getResultsOfQuery("SELECT y FROM " + dyadTable + " WHERE id=" + id);
		res_y.first();

		String serializedY = res_y.getString(1);
		String serializedX = res.getString(1);
		Double score = res.getDouble(2);

		double[] xArray = arrayDeserializer.splitAsStream(serializedX).mapToDouble(Double::parseDouble).toArray();
		double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble).toArray();

		Dyad dyad = new Dyad(new DenseDoubleVector(xArray), new DenseDoubleVector(yArray));
		return new Pair<Dyad, Double>(dyad, score);
	}

	/**
	 * Generates a {@link SparseDyadRankingInstance} in the following manner: <code>
	 * while there aren't enough dyads
	 *   collect all dyads with the specified dataset id
	 *   draw a random dyad using the seed
	 * sort the dyads
	 * return the sparse instance
	 * </code>
	 * 
	 * @return
	 * @throws SQLException
	 */
	private static SparseDyadRankingInstance getSparseDyadInstance(int datasetId, int seed, int length,
			SQLAdapter adapter) throws SQLException {
		// get all indices that have the correct dataset id
		// count the datasets
		ResultSet res = adapter
				.getResultsOfQuery("SELECT COUNT(id) FROM " + dyadTable + " WHERE dataset = " + datasetId);
		res.first();
		int indicesAmount = res.getInt(1);
		if (indicesAmount == 0)
			throw new IllegalArgumentException("No performance samples for for the dataset-id: " + datasetId);
		int[] dyadIndicesWithDataset = new int[indicesAmount];
		// collect the indices
		res = adapter.getResultsOfQuery("SELECT id FROM " + dyadTable + " WHERE dataset = " + datasetId);
		int counter = 0;
		while (res.next()) {
			dyadIndicesWithDataset[counter++] = res.getInt(1);
		}

		// now draw the dyads
		List<Pair<Dyad, Double>> dyads = new ArrayList<>(length);
		Random random = new Random(seed);

		for (int i = 0; i < length; i++) {
			int randomIndexOfArray = random.nextInt(indicesAmount);
			int randomIndexInDb = dyadIndicesWithDataset[randomIndexOfArray];
			dyads.add(getDyadAndScoreWithId(randomIndexInDb, adapter));
		}

		// sort the dyads and extract the sparse instance
		Vector singleX = dyads.iterator().next().getX().getInstance();
		List<Vector> sortedAlternatives = dyads.stream()
				.sorted((pair1, pair2) -> Double.compare(pair1.getY(), pair2.getY())).map(Pair::getX)
				.map(Dyad::getAlternative).collect(Collectors.toList());
		return new SparseDyadRankingInstance(singleX, sortedAlternatives);
	}

	public static DyadRankingDataset getSparseDyadDataset(int seed, int amountOfDyadInstances, int alternativeLength,
			int[] allowedDatasetsInSplit, SQLAdapter adapter) throws SQLException {

		List<IDyadRankingInstance> sparseDyadRankingInstances = new ArrayList<>();
		for (int i = 0; i < amountOfDyadInstances; i++) {
			int intermediateSeed = seed + i;
			int randomDataset = getRandomDatasetId(intermediateSeed, allowedDatasetsInSplit);
			sparseDyadRankingInstances
					.add(getSparseDyadInstance(randomDataset, intermediateSeed, alternativeLength, adapter));
		}
		return new DyadRankingDataset(sparseDyadRankingInstances);
	}

	public static void main(String... args) throws SQLException, TrainingException, PredictionException, IOException {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);

		DyadDatasetGeneratorConfig config = ConfigFactory.create(DyadDatasetGeneratorConfig.class);

		allowedDatasetIds = config.getDatasetIds().toArray(new Integer[config.getDatasetIds().size()]);

		dyadTable = config.getTableName();

		String resultTableName = "dyad_ranking_approach_1_results_standardize";
		createResultTable(adapter, resultTableName);

		int counter = 0;
		for (Integer trainRankingLength : config.getRankingLengthsTrainKey()) {

			for (Integer testRankingLength : config.getRankingLengthsTest()) {

				for (Integer trainRankingNum : config.getRankingNumTrain()) {

					for (Integer seed : config.getSeeds()) {

						TrainTestDatasetIds split = getTrainTestSplit(0.7d, seed);
						DyadRankingDataset trainDataset = getSparseDyadDataset(seed + 1, trainRankingNum,
								trainRankingLength, split.trainDatasetIds, adapter);
						DyadRankingDataset testDataset = getSparseDyadDataset(seed + 2, 200, testRankingLength,
								split.testDatasetIds, adapter);

						 AbstractDyadScaler scaler = new DyadUnitIntervalScaler();
						 scaler.fit(trainDataset);
						 scaler.transformAlternatives(trainDataset);
						 scaler.transformAlternatives(testDataset);

						PLNetDyadRanker ranker = new PLNetDyadRanker();

						System.out.println(ranker.getConfiguration().toString());

						ranker.train(trainDataset);
						ranker.saveModelToFile("out_no_scaler_cuda_" + counter);

						double loss = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(),
								testDataset, ranker);
						// x try (ObjectOutputStream osOut = new ObjectOutputStream(
						// new FileOutputStream(new File("scaler_out_" + counter + ".ser")))) {
						// osOut.writeObject(scaler);
						// } catch (IOException e) {
						// }

						System.out.println("Average Kendalls Tau: " + loss);
						Map<String, Object> results = new HashMap<>();
						results.put("id", counter);
						results.put("ranking_length_train", trainRankingLength);
						results.put("ranking_length_test", testRankingLength);
						results.put("avg_kendalls_tau", loss);
						results.put("seed", seed);
						results.put("num_dyads_train", trainRankingNum);
						try {
							adapter.insert(resultTableName, results);
						} catch (Exception e) {
							e.printStackTrace();
						}

						counter++;
					}
				}
			}
		}
		adapter.close();
	}

	private static void createResultTable(SQLAdapter adapter, String tableName) throws SQLException {
		ResultSet rs = adapter.getResultsOfQuery("SHOW TABLES");
		boolean hasPerformanceTable = false;
		while (rs.next()) {
			String ptableName = rs.getString(1);
			if (ptableName.equals(tableName))
				hasPerformanceTable = true;
		}

		if (!hasPerformanceTable) {
			adapter.update("CREATE TABLE " + tableName + " (`id` int(10) NOT NULL,\r\n"
					+ " `ranking_length_train` int(10) NOT NULL, \r \n" + " `ranking_length_test` int(10) NOT NULL,\r\n"
					+ " `avg_kendalls_tau` double NOT NULL, \r \n" + "`seed` int(10) NOT NULL, \r \n"
					+ "`num_dyads_train` int(10) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin",
					new ArrayList<>());
		}
	}

	@SuppressWarnings("unused")
	private static void writeDyadsAndScoresToFile(String filePath, List<Pair<Dyad, Double>> dyadScorePairs) {
		try (FileOutputStream out = new FileOutputStream(new File(DYAD_FILE))) {
			for (Pair<Dyad, Double> dyadScorePair : dyadScorePairs) {
				out.write(dyadScorePair.getX().getInstance().toString().getBytes());
				out.write(";".getBytes());
				out.write(dyadScorePair.getX().getAlternative().toString().getBytes());
				out.write("|".getBytes());
				out.write(dyadScorePair.getY().toString().getBytes());
				out.write("\n".getBytes());
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static TrainTestDatasetIds getTrainTestSplit(double ratio, int seed) {
		Random rand = new Random(seed);
		List<Integer> allDatasets = Arrays.stream(allowedDatasetIds).collect(Collectors.toList());
		Collections.shuffle(allDatasets, rand);
		int randomIndex = (int) Math.floor(allDatasets.size() * ratio);
		List<Integer> trainList = allDatasets.subList(0, randomIndex);
		List<Integer> testList = allDatasets.subList(randomIndex, allDatasets.size());
		TrainTestDatasetIds toReturn = new TrainTestDatasetIds();
		toReturn.trainDatasetIds = trainList.stream().mapToInt(i -> i).toArray();
		toReturn.testDatasetIds = testList.stream().mapToInt(i -> i).toArray();
		return toReturn;
	}

	private static int getRandomDatasetId(int intermediateSeed, int[] allowedDatasetsInSplit) {
		Random random = new Random(intermediateSeed);
		int index = random.nextInt(allowedDatasetsInSplit.length);
		return allowedDatasetsInSplit[index];
	}

	private static class TrainTestDatasetIds {
		int[] trainDatasetIds;
		int[] testDatasetIds;
	}

}
