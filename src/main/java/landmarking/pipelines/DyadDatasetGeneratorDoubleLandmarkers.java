package landmarking.pipelines;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.google.common.primitives.Doubles;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.dataset.SparseDyadRankingInstance;

public class DyadDatasetGeneratorDoubleLandmarkers {

	public static final int DATASET_IDS[] = { 44, 1462, 1063, 1480, 151, 1038, 333, 312, 334, 1510, 335, 50, 31, 37,
			1494, 1493, 1471, 1491, 1050, 1489, 1467, 1049, 3, 1487, 1068, 1046, 1464, 1067, 1504 };
	public static final int PIPELINE_IDS[] = { 392, 398, 596, 795, 872, 996, 557, 438, 913, 638, 54, 56, 760, 764, 843,
			967, 604, 209, 370, 1040, 570, 694, 255, 894, 1115, 455, 777, 657, 658, 816, 79, 1096, 262, 1092, 461, 264,
			782, 189, 783, 222, 465, 300, 1124, 269, 665, 864, 823, 309, 829 };
	public static final int PIPELINE_LANDMARKER_IDS[] = { 443, 52, 892, 172, 759, 865, 1079, 1109 };
	private static final Pattern arrayDeserializerBlanc = Pattern.compile(" ");

	public static void main(String args[]) {
		System.out.println("test");
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		DyadRankingDataset dataset = new DyadRankingDataset();
		HashMap<Integer, Vector> datasetMetafeatures = new HashMap<Integer, Vector>();
		HashMap<Integer, Vector> pipelineMetafeatures = new HashMap<Integer, Vector>();

		// load dataset metafeatures
		for (int datasetID : DATASET_IDS) {
			try {
				ResultSet rs = adapter.getResultsOfQuery(
						"SELECT dataset, X_LANDMARKERS FROM dataset_metafeatures_mirror WHERE dataset = " + datasetID);
				while (rs.next()) {
					String serializedX = rs.getString(2);
					double[] xArray = arrayDeserializerBlanc.splitAsStream(serializedX).mapToDouble(Double::parseDouble)
							.toArray();
					Vector features = new DenseDoubleVector(xArray);
					datasetMetafeatures.put(rs.getInt(1), features);
				}
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		// load pipeline metafeatures
		for (int pipelineID : PIPELINE_IDS) {
			try {
				ResultSet rs = adapter.getResultsOfQuery(
						"SELECT pipeline_id,  loss, dataset_id FROM pipeline_landmarking_v2 WHERE pipeline_id = "
								+ pipelineID + " ORDER BY dataset_id asc");
				ArrayList<Double> losses = new ArrayList<Double>();
				while (rs.next()) {
					double loss = rs.getDouble(2);
					losses.add(loss);
				}
				Vector features = new DenseDoubleVector(Doubles.toArray(losses));
				pipelineMetafeatures.put(pipelineID, features);
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

//		for (int datasetID : DATASET_IDS) {
//			List<Pair<Integer, Double>> idScorePairs = new ArrayList<Pair<Integer, Double>>();
//			for (int pipelineID : PIPELINE_IDS) {
//				try {
//					ResultSet rs = adapter.getResultsOfQuery(
//							"SELECT dataset_id, pipeline_id, loss FROM draco_pipeline_performance WHERE dataset_id = "
//									+ datasetID + " AND pipeline_ID = " + pipelineID);
//					while (rs.next()) {
//						idScorePairs.add(new Pair<Integer, Double>(rs.getInt(2), rs.getDouble(3)));
//					}
//				} catch (SQLException e) {
//					// TODO Auto-generated catch block
//					e.printStackTrace();
//				}
//
//			}
//			Collections.sort(idScorePairs, Comparator.comparing(p -> p.getRight()));
//			Vector instance = datasetMetafeatures.get(datasetID);
//			List<Vector> alternatives = new ArrayList<Vector>();
//			for (Pair<Integer, Double> pair : idScorePairs) {
//				alternatives.add(pipelineMetafeatures.get(pair.getLeft()));
////				System.out.println(
////						"dataset: " + datasetID + " pipeline: " + pair.getLeft() + " loss: " + pair.getRight());
////				System.out.println(
////						"pipeline" + pair.getLeft() + " meta: " + " \t " + pipelineMetafeatures.get(pair.getLeft()));
////				System.out.println("dataset" + datasetID + " meta: " + " \t " + datasetMetafeatures.get(datasetID));
//			}
//			SparseDyadRankingInstance drInstance = new SparseDyadRankingInstance(instance, alternatives);
//			dataset.add(drInstance);
//			try {
//				FileOutputStream fos = new FileOutputStream(new File("landmarking_data_v2.txt"));
//				dataset.serialize(fos);
//			} catch (FileNotFoundException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}
//
//		}
		
		List<Integer> pIDS = Arrays.stream(PIPELINE_IDS).boxed().collect(Collectors.toList());
		List<Integer> dIDS = Arrays.stream(DATASET_IDS).boxed().collect(Collectors.toList());
		
		DyadRankingDataset ds = new DyadRankingDataset();
		HashMap<Integer, ArrayList<Integer>> orderings = new HashMap<Integer, ArrayList<Integer>>();
		try {
			ResultSet reSet = adapter.getResultsOfQuery(
					"SELECT dataset, pipeline_id, X_LANDMARKERS, pipeMeta, loss FROM dataset_metafeatures_mirror NATURAL JOIN (SELECT dataset_id as dataset, loss, pipeMeta, pipeline_id FROM (SELECT pipeline_id, COUNT(dataset_id) as cnt, GROUP_CONCAT(loss ORDER BY dataset_id DESC) as pipeMeta FROM `pipeline_landmarking_v2` GROUP BY pipeline_id) AS T NATURAL JOIN draco_pipeline_performance) AS NT ORDER BY dataset, loss");
			while(reSet.next()) {
				int dId = reSet.getInt(1);
				int pId = reSet.getInt(2);
				if(pIDS.contains(pId) && dIDS.contains(dId)) {
				//				System.out.println(reSet.getInt(1) + " " + reSet.getInt(2) + " " + reSet.getString(4) + " " + reSet.getDouble(5));
				if(!orderings.containsKey(dId))
					orderings.put(dId, new ArrayList<Integer>());
				orderings.get(dId).add(pId);
				}
			}
		} catch (SQLException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		System.out.println(orderings);
		
		for(int dID : orderings.keySet()) {
			ArrayList<Integer> pipelines = orderings.get(dID);
			ArrayList<Vector> alternatives = new ArrayList<Vector>();
			Vector instance = datasetMetafeatures.get(dID);
			for(int pID : pipelines) {
				alternatives.add(pipelineMetafeatures.get(pID));
			}
			IDyadRankingInstance drInstance = new SparseDyadRankingInstance(instance, alternatives);
			ds.add(drInstance);
		}
		System.out.println(ds);
		try {
			FileOutputStream fos = new FileOutputStream(new File("double_landmarking_dataset.txt"));
			ds.serialize(fos);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
//		System.out.println(dataset);
	}

}