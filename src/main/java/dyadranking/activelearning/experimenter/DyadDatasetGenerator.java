package dyadranking.activelearning.experimenter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;

public class DyadDatasetGenerator {

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final Pattern arrayDeserializer = Pattern.compile(" ");

	private static String X_KEY = "X_LANDMARKERS";

	private static List<Integer> allDatasets = Arrays.asList(44, 1462, 1063, 1480, 151, 1038, 333, 312, 334, 1510, 335,
			50, 31, 37, 1494, 1493, 1471, 1491, 1050, 1489, 1467, 1049, 3, 1487, 1068, 1046, 1464, 1067, 1504);

//	private static List<Integer> allDatasets = Arrays.asList(44);

	public static void main(String args[]) {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);

		DyadRankingDataset drDataset = new DyadRankingDataset();

		for (Integer datasetId : allDatasets) {
			List<Pair<Dyad, Double>> allDyads = new ArrayList<Pair<Dyad, Double>>();
			ResultSet res_y;
			HashSet<Dyad> seenDyads = new HashSet<Dyad>();
			HashSet<Double> seenScores = new HashSet<Double>();
			try {
				res_y = adapter.getResultsOfQuery("SELECT y, score, " + X_KEY + " , pipeline_id FROM " + dyadTable
						+ " NATURAL JOIN " + datasetMetaFeatureTable + " WHERE dataset=" + datasetId.toString());
				while (res_y.next()) {
					if (res_y.getInt(4) % 20 == 0) {
						String serializedY = res_y.getString(1);
						Double score = res_y.getDouble(2);
						String serializedX = res_y.getString(3);
						double[] xArray = arrayDeserializer.splitAsStream(serializedX).mapToDouble(Double::parseDouble)
								.toArray();
						double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble)
								.toArray();
						Dyad dyad = new Dyad(new DenseDoubleVector(xArray), new DenseDoubleVector(yArray));
						if (!seenScores.contains(score) && !seenDyads.contains(dyad)) {
							allDyads.add(new Pair<Dyad, Double>(dyad, score));
							seenScores.add(score);
							seenDyads.add(dyad);
						}
					}
				}
				DyadRankingInstance instance = new DyadRankingInstance(
						allDyads.stream().sorted((pair1, pair2) -> Double.compare(pair1.getY(), pair2.getY()))
								.map(Pair::getX).collect(Collectors.toList()));
				drDataset.add(instance);
				List<Pair<Dyad, Double>> dyadListWithScore = allDyads.stream().sorted(
						(Pair<Dyad, Double> pair1, Pair<Dyad, Double> pair2) -> pair1.getY().compareTo(pair2.getY()))
						.collect(Collectors.toList());
				for (Pair<Dyad, Double> pair : dyadListWithScore)
					System.out.println(pair);
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		try {
			drDataset.serialize(new FileOutputStream(new File("./output_dataset.txt")));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}