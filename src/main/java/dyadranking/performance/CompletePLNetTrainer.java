package dyadranking.performance;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.util.DyadMinMaxScaler;

/**
 * Trains and serializes a PLNet with all dyads available.
 *
 */
public class CompletePLNetTrainer {

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final Pattern arrayDeserializer = Pattern.compile(" ");

	private static String X_KEY = "X_LANDMARKERS";

	private static List<Integer> allDatasets = Arrays.asList(44, 1462, 1063, 1480, 151, 1038, 333, 312, 334, 1510, 335,
			50, 31, 37, 1494, 1493, 1471, 1491, 1050, 1489, 1467, 1049, 3, 1487, 1068, 1046, 1464, 1067, 1504);

	public static void main(String[] args) throws SQLException, TrainingException, PredictionException, IOException {
	//	SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);


//		double totalKenndallsTau__noscale = 0.0d;
//		double totalKendallsTau__normalize = 0.0d;
//		// do a leave one out cross validation
//		for (int i = 0; i < allDatasets.size(); i++) {
//			System.out.println("Leaving out " + allDatasets.get(i));
//			List<Integer> copy = new ArrayList<>(allDatasets);
//			copy.remove(i);
//			System.out.println("Starting no_scale calculation...");
//			List<IDyadRankingInstance> train = getCompleteDyadRanking(adapter, copy);
//			List<IDyadRankingInstance> test = getCompleteDyadRanking(adapter, Arrays.asList(algorithmIndeces.get(i)));
//			PLNetDyadRanker ranker = new PLNetDyadRanker();
//
//			DyadRankingDataset trainDS = new DyadRankingDataset(train);
//			DyadRankingDataset testDS = new DyadRankingDataset(test);
//
//			// ranker.train(trainDS);
//
//			// double kTau = DyadRankingLossUtil.computeAverageLoss(new
//			// KendallsTauDyadRankingLoss(), testDS, ranker);
//			// totalKenndallsTau__noscale += kTau;
//			// System.out.println("Kendalls Tau is " + kTau);
//			System.out.println("Now with normalization...");
//
//			DyadMinMaxScaler scaler = new DyadMinMaxScaler();
//
//			scaler.fit(trainDS);
//			scaler.transformAlternatives(trainDS, algorithmIndeces);
//			scaler.transformAlternatives(testDS, algorithmIndeces);
//
//			ranker = new PLNetDyadRanker();
//			ranker.train(trainDS);
//			double kTau = DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), testDS, ranker);
//			System.out.println("Kendalls Tau is " + kTau);
//			totalKendallsTau__normalize += kTau;
//		}
//
//		double finalKTau__noscale = totalKenndallsTau__noscale / allDatasets.size();
//		System.out.println("Final KTAU no scale: " + finalKTau__noscale);
//
//		double finalKTau__normalize = totalKendallsTau__normalize / allDatasets.size();
//		System.out.println("Final KTAU no scale: " + finalKTau__normalize);

		System.out.println("Final training!");
		System.out.println("Starting with no_scale");
		DyadRankingDataset trainDS  = getCompleteDyadRanking(null, allDatasets);

		PLNetDyadRanker ranker = new PLNetDyadRanker();
		ranker.train(trainDS);
		System.out.println("Saving to File");
		ranker.saveModelToFile("final_plnet_noscale");

		System.out.println("Now with minmax scaling");
		DyadMinMaxScaler scaler = new DyadMinMaxScaler();
		scaler.fit(trainDS);
		scaler.transformAlternatives(trainDS);
		ranker = new PLNetDyadRanker();
		ranker.train(trainDS);
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("minmaxscaler.ser"));
		oos.writeObject(scaler);
		oos.close();

		ranker.saveModelToFile("final_plnet_minmax");
	}

	/**
	 * Queries the DB to extract the dyad and its' perfomance score.
	 * 
	 * @param id
	 *            specifies the db entry
	 * @return the dyad
	 * @throws SQLException
	 * @throws FileNotFoundException 
	 */
	private static DyadRankingDataset getCompleteDyadRanking(SQLAdapter adapter, List<Integer> datasetIds)
			throws SQLException, FileNotFoundException {

		List<DyadRankingDataset> allDatasets = new ArrayList<>();
		for (int dataset : datasetIds) { 
			DyadRankingDataset drSet = new DyadRankingDataset();
			drSet.deserialize(new FileInputStream("context_sensitive_dataset_id_" + dataset+".dataset"));
			allDatasets.add(drSet);
		}
		return null;
		//	return DyadRankingDataset.mergeDatasets(allDatasets.toArray(new DyadRankingDataset [allDatasets.size()]));
	}

	public List<IDyadRankingInstance> getContextSensitivePairwiseRanking(SQLAdapter adapter, List<Integer> datasetIds)
			throws SQLException {
		List<IDyadRankingInstance> allPairs = new ArrayList<>();

		for (Integer datasetId : datasetIds) {
			List<Pair<Double, Dyad>> allDyads = new ArrayList<>();
			ResultSet res_y = adapter.getResultsOfQuery("SELECT y, score, " + X_KEY + " FROM " + dyadTable
					+ " NATURAL JOIN " + datasetMetaFeatureTable + " WHERE dataset=" + datasetId.toString());
			while (res_y.next()) {

				String serializedY = res_y.getString(1);
				Double score = res_y.getDouble(2);
				String serializedX = res_y.getString(3);
				double[] xArray = arrayDeserializer.splitAsStream(serializedX).mapToDouble(Double::parseDouble)
						.toArray();

				double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble)
						.toArray();
				allDyads.add(new Pair<>(score, new Dyad(new DenseDoubleVector(xArray), new DenseDoubleVector(yArray))));
			}
			for (int i = 0; i < allDyads.size(); i++) {
				for (int j = i; j < allDyads.size(); j++) {
					Pair<Double, Dyad> dyadA = allDyads.get(i);
					Pair<Double, Dyad> dyadB = allDyads.get(j);
					List<Pair<Double, Dyad>> dyads = new ArrayList<>();
					dyads.add(dyadA);
					dyads.add(dyadB);
					DyadRankingInstance instance = new DyadRankingInstance(
							dyads.stream().sorted((pair1, pair2) -> Double.compare(pair1.getX(), pair1.getX()))
									.map(Pair::getY).collect(Collectors.toList()));
					allPairs.add(instance);
				}
			}
		}
		return allPairs;
	}

}
