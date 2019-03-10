package dyadranking.performance;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.util.DyadMinMaxScaler;

/**
 * For each dataset it draws 10000 pairs uniformly at random without
 * replacement. Then, it shuffles all of these pairs and trains PLNet.
 * 
 * @author mirko
 *
 */
public class SubsamplingBasedPLNetTrainer {

	private static final Pattern arrayDeserializer = Pattern.compile(" "); 
	
	private static String dyadTable = "dyad_dataset_approach_5_performance_samples_full";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final String metaFeatureName = "X_LANDMARKERS";

	private static Random random = new Random(42);

	
	private static List<Integer> allDatasets = Arrays.asList(44, 1462, 1063, 1480, 151, 1038, 333, 312, 334, 1510, 335,
			50, 31, 37, 1494, 1493, 1471, 1491, 1050, 1489, 1467, 1049, 3, 1487, 1068, 1046, 1464, 1067, 1504);


	public static void main(String[] args) throws Exception {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
	//	List<List<IDyadRankingInstance>> allLists = new ArrayList<>();
//		for (int datasetId : allDatasets) {
//			allLists.add(getSubsamplesForDatasetId(100000, datasetId, adapter));
//			System.out.println("Done with "+ datasetId);
//		}
//		List<IDyadRankingInstance> completeRanking = allLists.stream().flatMap(List::stream).collect(Collectors.toList());
//		Collections.shuffle(completeRanking, random);
		DyadRankingDataset dataset = new DyadRankingDataset();
		dataset.deserialize(new FileInputStream(args[4]));
		//dataset.serialize(new FileOutputStream("subsampled_dataset.dataset"));
		System.out.println("Got "+ args[4]);

//		System.out.println("Final training!");
//		System.out.println("Starting with no_scale");
//
//		PLNetDyadRanker ranker = new PLNetDyadRanker();
//		ranker.train(dataset);
//		System.out.println("Saving to File");
//		ranker.saveModelToFile("final_plnet_noscale");

		System.out.println("Now with minmax scaling");
		DyadMinMaxScaler scaler = new DyadMinMaxScaler();
		scaler.fit(dataset);
		scaler.transformAlternatives(dataset);
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(args[4]+"_minmaxscaler.ser"));
		oos.writeObject(scaler);
		oos.close();
		PLNetDyadRanker ranker = new PLNetDyadRanker();
		ranker.train(dataset);
		ranker.saveModelToFile(args[4]+"_final_plnet_minmax");
	}

	private static List<IDyadRankingInstance> getSubsamplesForDatasetId(int length, int datasetId, SQLAdapter adapter)
			throws SQLException {
		List<IDyadRankingInstance> toReturn = new ArrayList<>();
		ResultSet resSet = adapter.getResultsOfQuery("SELECT id FROM " + dyadTable + " WHERE dataset=" + datasetId);
		List<Integer> allIdsForDataset = new ArrayList<>();

		while (resSet.next()) {
			allIdsForDataset.add(resSet.getInt(1));
		}

		// now draw random entries but keep track of pairs we already have drawn
		List<Pair<Integer, Integer>> alldrawnPairs = new ArrayList<>();

		while (toReturn.size() < length) {

			// draw first index
			int firstIndex = random.nextInt(allIdsForDataset.size());
			int firstDyadTableIndex = allIdsForDataset.get(firstIndex);

			// draw second index
			
			int secondIndex = random.nextInt(allIdsForDataset.size());
			while (secondIndex == firstIndex) {
				secondIndex = random.nextInt(allIdsForDataset.size());
			}
			int secondDyadTableIndex = allIdsForDataset.get(secondIndex);

			Pair<Integer, Integer> pair;

			// order indices to maintain comparability
			if (firstDyadTableIndex < secondDyadTableIndex) {
				pair = new Pair<Integer, Integer>(firstDyadTableIndex, secondDyadTableIndex);
			} else {
				pair = new Pair<Integer, Integer>(secondDyadTableIndex, firstDyadTableIndex);
			}

			if (alldrawnPairs.contains(pair)) {
				System.out.println("Duplicate pair found, drawing another one");
				continue;
			}

			// now extract the dyads
			ResultSet resDyads = adapter.getResultsOfQuery("SELECT y, score, " + metaFeatureName + " FROM " + dyadTable
					+ " NATURAL JOIN " + datasetMetaFeatureTable + " WHERE id IN (" + firstDyadTableIndex + ", "
					+ secondDyadTableIndex + ")");
			
			resDyads.first();
			// create first dyad
			Pair<Double, Dyad> firstDyad = getDyadFromResultSet(resDyads);
			resDyads.next();
			Pair<Double, Dyad> secondDyad = getDyadFromResultSet(resDyads);

			if (firstDyad.getX().equals(secondDyad.getX())) {
				System.out.println("Score too similar");
				continue;
			}
			
			// we are now sure that we want this pair
			alldrawnPairs.add(pair);
			
			if (Double.compare(firstDyad.getX(), secondDyad.getX()) < 0) {
				toReturn.add(new DyadRankingInstance(Arrays.asList(firstDyad.getY(), secondDyad.getY())));
			} else {
				toReturn.add(new DyadRankingInstance(Arrays.asList(secondDyad.getY(), firstDyad.getY())));
			}
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
}
