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
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;

public class DyadDatasetGenerator {

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final Pattern arrayDeserializer = Pattern.compile(" ");

	private static String X_KEY = "X_LANDMARKERS";

//	private static List<Integer> allDatasets = Arrays.asList(44, 1462, 1063, 1480, 151, 1038, 333, 312, 334, 1510, 335,
//			50, 31, 37, 1494, 1493, 1471, 1491, 1050, 1489, 1467, 1049, 3, 1487, 1068, 1046, 1464, 1067, 1504);

	private static List<Integer> allDatasets = Arrays.asList(44, 1462, 1063, 1480, 151, 1038, 333, 312, 334, 1510, 335,
			50, 31);

	private static List<Integer> allowedPipelines = Arrays.asList(3327, 3309, 2586, 2548, 785, 2280, 708, 572, 2896,
			2270, 3249, 2571, 3444, 3050, 309, 2, 1553, 2759, 2114, 2509, 2592, 3144, 3398, 2696, 531, 1205, 1783, 2131,
			2978, 2989, 2275, 1065, 729, 1235, 3368, 1157, 1816, 2713, 2917, 2093, 2849, 1887, 1154, 212, 3373, 728,
			1385, 3102, 3369, 498, 515, 692, 2359, 3292, 2864, 558, 3135, 3349, 2619, 2630, 2768, 158, 1017, 3381, 1575,
			2633, 2437, 1145, 3002, 3319, 2828, 278, 892, 3476, 1037, 1645, 3362, 2928, 3043, 2540, 338, 2622, 794,
			1818, 3229, 536, 2986, 1092, 2210, 2175, 470, 1322, 252, 1373, 730, 1750, 1240, 1461, 685, 1667, 1026, 2527,
			390, 929, 3203, 932, 546, 2611, 1347, 70, 2572, 1993, 3407, 2541, 287, 2841, 2478, 1842, 2949, 1047, 2591,
			2969, 2594, 706, 1394, 2677, 2546, 1307, 1874, 3033, 1781, 2149, 102, 1119, 860, 3412, 621, 2200, 2935,
			3161, 611, 3013, 3006, 703, 3218, 557, 3001, 3237, 238, 651, 1238, 1267, 1057, 2605, 2957, 1054, 2411, 823,
			1446, 2057, 1825, 949, 770, 2566, 1418, 1701, 405, 2753, 1420, 1534, 3066, 3387, 589, 2152, 1547, 1761, 911,
			3244, 798, 2788, 2877, 3284, 3070, 3461, 1409, 3056, 1332, 3264, 186, 1647, 2561, 2692, 1063, 2553, 2822,
			3314, 491, 2126, 2536, 2184, 3371, 2648, 1838, 3000, 396, 2113, 2635, 2742, 1508, 368, 1191, 214, 134, 1678,
			2735, 2375, 838, 846, 3367, 1987, 891, 2890, 3483, 2163, 2185, 2705, 1337, 707, 301, 777, 2914, 1952, 1546,
			1196, 112, 3157, 1393, 2338, 1022, 1215, 21, 1760, 105, 683, 2428, 3219, 2517, 939, 1336, 2845, 584, 1259,
			795, 2043, 2760, 1025, 690, 3313, 451, 2417, 3445, 827, 447, 963, 2745, 1782, 1797, 318, 3199, 269, 3338,
			642, 2324, 1340, 3234, 3320, 1144, 734, 2374, 121, 1352, 1607, 204, 1142, 1159, 743, 1892, 575, 510, 2559,
			410, 2555, 2584, 807, 1405, 3448, 2323, 2036, 1543, 3247, 1223, 3395, 243, 2447, 3388, 1146, 788, 3061, 1,
			168, 581, 1882, 3339, 2094, 634, 623, 980, 128, 197, 1372, 2173, 1677, 3110, 2395, 1626, 3274, 592, 3225,
			2122, 957, 3507, 2840, 1636, 2083, 155, 1861, 375, 924, 523, 1098, 632, 3183, 898, 1059, 2684, 1041, 1309,
			2068, 2714, 429, 2668, 24, 163, 2538, 1889, 802, 2985, 1258, 67);

//	private static List<Integer> allDatasets = Arrays.asList(44);

	public static void main(String args[]) {
		System.out.println(allowedPipelines.size());
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);

		DyadRankingDataset drDataset = new DyadRankingDataset();
		HashSet<Integer> pipelineIDsWithUniqueScores = new HashSet<Integer>(allowedPipelines);
		for (Integer datasetId : allDatasets) {
			List<Pair<Dyad, Double>> allDyads = new ArrayList<Pair<Dyad, Double>>();
			ResultSet res_y;
			HashSet<Dyad> seenDyads = new HashSet<Dyad>();
			HashSet<Double> seenScores = new HashSet<Double>();
			try {
				res_y = adapter.getResultsOfQuery("SELECT y, score, " + X_KEY + " , pipeline_id FROM " + dyadTable
						+ " NATURAL JOIN " + datasetMetaFeatureTable + " WHERE dataset=" + datasetId.toString());
				while (res_y.next()) {
					int pipelineID = res_y.getInt(4);
					if (allowedPipelines.contains(pipelineID)) {
						String serializedY = res_y.getString(1);
						Double score = res_y.getDouble(2);
						String serializedX = res_y.getString(3);
						double[] xArray = arrayDeserializer.splitAsStream(serializedX).mapToDouble(Double::parseDouble)
								.toArray();
						double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble)
								.toArray();
						Dyad dyad = new Dyad(new DenseDoubleVector(xArray), new DenseDoubleVector(yArray));
						if (seenScores.contains(score)) {
							System.out.println("Score already seen!");
							pipelineIDsWithUniqueScores.remove(pipelineID);
						} else if (seenDyads.contains(dyad)) {
							System.out.println("Dyad already seen!");
						} else {
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
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		for (IInstance inst : drDataset) {
			System.out.println("Ranking length: " + ((IDyadRankingInstance) inst).length());
			System.out.println("pipeline ids with unique scores size: " + pipelineIDsWithUniqueScores.size());
			System.out.println("pipeline ids with unique scores: " + pipelineIDsWithUniqueScores);
		}
		try {
			drDataset.serialize(new FileOutputStream(new File("./output_dataset.txt")));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}