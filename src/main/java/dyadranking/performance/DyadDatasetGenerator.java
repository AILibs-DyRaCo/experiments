package dyadranking.performance;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.dataset.SparseDyadRankingInstance;
import jaicore.ml.dyadranking.util.DyadStandardScaler;

public class DyadDatasetGenerator {

	private static final int DATASET_COUNT = 3421;

	private static final int[] allowedDatasetIds = { 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 28, 3, 30, 32, 36, 38, 44,
			46, 5, 6 };

	private static final String dyadTable = "dyad_dataset_new";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";
	
	private static final Pattern arrayDeserializer = Pattern.compile(" ");

	private static final String DYAD_FILE = "dyad_pool.txt";

	private static List<Pair<Dyad, Double>> dyadsWithScore = new LinkedList<Pair<Dyad, Double>>();

	private enum X_METRIC {
		X_LANDMARKERS("X_LANDMARKERS"), X_OTHERS("X_OTHERS");

		X_METRIC(String dbColumn) {
			this.dbColumn = dbColumn;
		}

		protected String dbColumn;

		@Override
		public String toString() {
			return dbColumn;
		}
	}

	private static X_METRIC xMetric = X_METRIC.X_LANDMARKERS;

	/**
	 * Queries the DB to extract the dyad and its' perfomance score.
	 * 
	 * @param id
	 *            specifies the db entry
	 * @return the dyad
	 * @throws SQLException
	 */
	private static Pair<Dyad, Double> getDyadAndScoreWithId(int id, SQLAdapter adapter) throws SQLException {
		ResultSet res = adapter.getResultsOfQuery("SELECT " + xMetric + ", score FROM "+dyadTable+" NATURAL JOIN "+datasetMetaFeatureTable+" WHERE id=" + id);
		if (res.wasNull())
			throw new IllegalArgumentException("No entry with id " + id);

		res.first();

		ResultSet res_y = adapter.getResultsOfQuery("SELECT y FROM "+dyadTable+" WHERE id=" + id);
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
	 * Gets all dyads and scores from the database.
	 * 
	 * @return the list of all dyads with the corresponding score
	 * @throws SQLException
	 */
	private static List<Pair<Dyad, Double>> getAllDyadsAndScores(SQLAdapter adapter) throws SQLException {

		ResultSet res = adapter.getResultsOfQuery(
				"SELECT " + xMetric + ", y" + ", score FROM " + dyadTable + " NATURAL JOIN " + datasetMetaFeatureTable);
		res.first();
		List<Pair<Dyad, Double>> dyadScoreList = new LinkedList<Pair<Dyad, Double>>();

//		ResultSet res_y = adapter.getResultsOfQuery("SELECT y FROM " + dyadTable);
//		res_y.first();

		while (res.next()) {
			
			String serializedX = res.getString(1);
			String serializedY = res.getString(2);
			Double score = res.getDouble(3);

			double[] xArray = arrayDeserializer.splitAsStream(serializedX).mapToDouble(Double::parseDouble).toArray();
			double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble).toArray();
			
			Dyad dyad = new Dyad(new DenseDoubleVector(xArray), new DenseDoubleVector(yArray));
			dyadScoreList.add(new Pair<Dyad, Double>(dyad, score));

		}

		return dyadScoreList;
	}

	/**
	 * Generates a {@link DyadRankingInstance} in the following manner: <code>
	 * while there aren't enough dyads
	 *   draw a random dyad using the seed
	 * sort the dyads
	 * return the dyad ranking instance
	 * </code>
	 * 
	 * @return
	 * @throws SQLException
	 */
	private static DyadRankingInstance getDyadInstanceForSeedAndLength(int seed, int length, SQLAdapter adapter)
			throws SQLException {
		// now draw the dyads
		List<Pair<Dyad, Double>> dyads = new ArrayList<>(length);
		Random random = new Random(seed);
		for (int i = 0; i < length; i++) {
			int randomDyadIndex = random.nextInt(DATASET_COUNT) + 1;
			dyads.add(getDyadAndScoreWithId(randomDyadIndex, adapter));
		}

		// sort the dyads and extract the sparse instance

		List<Dyad> sortedDyads = dyads.stream().sorted((pair1, pair2) -> Double.compare(pair1.getY(), pair2.getY()))
				.map(Pair::getX).collect(Collectors.toList());
		return new DyadRankingInstance(sortedDyads);
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

	public static DyadRankingDataset getDyadDataset(int seed, int amountOfDyadInstances, int alternativeLength,
			SQLAdapter adapter) throws SQLException {

		List<IDyadRankingInstance> sparseDyadRankingInstances = new ArrayList<>();
		for (int i = 0; i < amountOfDyadInstances; i++) {
			int intermediateSeed = seed + i;
			sparseDyadRankingInstances
					.add(getDyadInstanceForSeedAndLength(intermediateSeed, alternativeLength, adapter));
		}
		return new DyadRankingDataset(sparseDyadRankingInstances);
	}

	public static void main(String... args) throws SQLException, TrainingException, IOException {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		TrainTestDatasetIds split = getTrainTestSplit(0.7d);
		DyadRankingDataset trainDataset = getSparseDyadDataset(42, 3000, 10, split.trainDatasetIds, adapter);
	//	DyadRankingDataset testDataset = getSparseDyadDataset(43, 200, 10, split.testDatasetIds, adapter);
		
		DyadStandardScaler scaler = new DyadStandardScaler();
		scaler.fit(trainDataset);
		scaler.printMeans();
		scaler.printStandardDeviations();
//		scaler.transformInstances(trainDataset);
//		scaler.transformInstances(testDataset);
		
		PLNetDyadRanker ranker = new PLNetDyadRanker();
		System.out.println(ranker.getConfiguration().toString());
		ranker.train(trainDataset);
		ranker.saveModelToFile("out");
//		try {
//			double loss = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), testDataset, ranker);
//			System.out.println("Average Kendalls Tau: " + loss);
//		} catch (PredictionException e) {
//			e.printStackTrace();
//		}
		adapter.close();
	}
	
	private static void writeDyadsAndScoresToFile(String filePath, List<Pair<Dyad, Double>> dyadScorePairs) {
		FileOutputStream out;
		try {
			out = new FileOutputStream(new File(DYAD_FILE));
			for (Pair<Dyad, Double> dyadScorePair : dyadScorePairs) {
				out.write(dyadScorePair.getX().getInstance().toString().getBytes());
				out.write(";".getBytes());
				out.write(dyadScorePair.getX().getAlternative().toString().getBytes());
				out.write("|".getBytes());
				out.write(dyadScorePair.getY().toString().getBytes());
				out.write("\n".getBytes());
			}
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	

	private static TrainTestDatasetIds getTrainTestSplit(double ratio) {
		List<Integer> allDatasets = Arrays.stream(allowedDatasetIds).mapToObj(i -> (Integer) i)
				.collect(Collectors.toList());
		Collections.shuffle(allDatasets);
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
