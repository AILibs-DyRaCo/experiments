package dyadranking.performance;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import dyadranking.sql.SQLUtils;
import hasco.model.ComponentInstance;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;

/**
 * Trains and serializes a PLNet with all dyads available.
 *
 */
public class ContextSensitiveAllPairsDyadDatasetGen {

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final Pattern arrayDeserializer = Pattern.compile(" ");

	private static String X_KEY = "X_LANDMARKERS";

	private static List<Integer> allDatasets = Arrays.asList(44, 1462, 1063, 1480, 151, 1038, 333, 312, 334, 1510, 335,
			50, 31, 37, 1494, 1493, 1471, 1491, 1050, 1489, 1467, 1049, 3, 1487, 1068, 1046, 1464, 1067, 1504);

	public static void main(String[] args) throws SQLException, TrainingException, PredictionException, IOException {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);

		// double totalKenndallsTau__noscale = 0.0d;
		// double totalKendallsTau__normalize = 0.0d;
		// // do a leave one out cross validation
		// for (int i = 0; i < allDatasets.size(); i++) {
		// System.out.println("Leaving out " + allDatasets.get(i));
		// List<Integer> copy = new ArrayList<>(allDatasets);
		// copy.remove(i);
		// System.out.println("Starting no_scale calculation...");
		// List<IDyadRankingInstance> train = getCompleteDyadRanking(adapter, copy);
		// List<IDyadRankingInstance> test = getCompleteDyadRanking(adapter,
		// Arrays.asList(algorithmIndeces.get(i)));
		// PLNetDyadRanker ranker = new PLNetDyadRanker();
		//
		// DyadRankingDataset trainDS = new DyadRankingDataset(train);
		// DyadRankingDataset testDS = new DyadRankingDataset(test);
		//
		// // ranker.train(trainDS);
		//
		// // double kTau = DyadRankingLossUtil.computeAverageLoss(new
		// // KendallsTauDyadRankingLoss(), testDS, ranker);
		// // totalKenndallsTau__noscale += kTau;
		// // System.out.println("Kendalls Tau is " + kTau);
		// System.out.println("Now with normalization...");
		//
		// DyadMinMaxScaler scaler = new DyadMinMaxScaler();
		//
		// scaler.fit(trainDS);
		// scaler.transformAlternatives(trainDS, algorithmIndeces);
		// scaler.transformAlternatives(testDS, algorithmIndeces);
		//
		// ranker = new PLNetDyadRanker();
		// ranker.train(trainDS);
		// double kTau = DyadRankingLossUtil.computeAverageLoss(new
		// KendallsTauDyadRankingLoss(), testDS, ranker);
		// System.out.println("Kendalls Tau is " + kTau);
		// totalKendallsTau__normalize += kTau;
		// }
		//
		// double finalKTau__noscale = totalKenndallsTau__noscale / allDatasets.size();
		// System.out.println("Final KTAU no scale: " + finalKTau__noscale);
		//
		// double finalKTau__normalize = totalKendallsTau__normalize /
		// allDatasets.size();
		// System.out.println("Final KTAU no scale: " + finalKTau__normalize);
		getContextSensitivePairwiseRanking(adapter, allDatasets);

	}

	public static void getContextSensitivePairwiseRanking(SQLAdapter adapter, List<Integer> datasetIds)
			throws SQLException {
		Executor executor = Executors.newFixedThreadPool(6);
		CompletionService<DyadRankingDataset> completionService = new ExecutorCompletionService<>(executor);

		for (final Integer datasetId : datasetIds) {
			completionService.submit(() -> {
				System.out.println("Now dataset " + datasetId);
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
					allDyads.add(
							new Pair<>(score, new Dyad(new DenseDoubleVector(xArray), new DenseDoubleVector(yArray))));
				}
				List<IDyadRankingInstance> allPairs = new ArrayList<>();
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
				DyadRankingDataset dataset = new DyadRankingDataset(allPairs);
				dataset.serialize(new FileOutputStream("context_sensitive_dataset_id_" + datasetId+".dataset"));
				return dataset;
			});
		}
		for (int i = 0; i < datasetIds.size(); i++) {
			try {
				completionService.take().get();
			} catch (InterruptedException | ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
	}

}
