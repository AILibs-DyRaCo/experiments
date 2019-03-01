package dyadranking.performancebasedpipelines;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import com.google.common.collect.Lists;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import jaicore.basic.SQLAdapter;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.dataset.SparseDyadRankingInstance;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;
import jaicore.ml.metafeatures.RandomTreePerformanceBasedFeatureGenerator;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class PerformanceBasedPipelinesExperiments {

	public static void main(String[] args) throws Exception {
		SQLAdapter adapter = new SQLAdapter(args[0], args[1], args[2], args[3]);

		createTable(adapter);
	}
	
	public static void evaluate(SQLAdapter adapter) throws SQLException, PredictionException, TrainingException {
		DyadRankingDataset dataset = createDyadRankingDatasetFromTable(adapter);

		KendallsTauDyadRankingLoss lossFunction = new KendallsTauDyadRankingLoss();
		double loss = 0;
		System.out.println("Compute Loss for Dataset: " + dataset.size());
		for (int i = 0; i < dataset.size(); i++) {
			SparseDyadRankingInstance actual = (SparseDyadRankingInstance) dataset.remove(i);
			List<Dyad> shuffleContainer = Lists.newArrayList(actual.iterator());
			Collections.shuffle(shuffleContainer);
			IDyadRankingInstance shuffledActual = new DyadRankingInstance(shuffleContainer);

			PLNetDyadRanker ranker = new PLNetDyadRanker();
			ranker.train(dataset);
			double currentLoss = lossFunction.loss(actual, ranker.predict(shuffledActual));
			System.out.println(currentLoss);
			loss += currentLoss;

			dataset.add(i, actual);
		}
		loss = loss / dataset.size();
		System.out.println(loss);
	}

	public static DyadRankingDataset createDyadRankingDatasetFromTable(SQLAdapter adapter) throws SQLException {
		DyadRankingDataset dataset = new DyadRankingDataset();

		ResultSet resultSet = adapter.getResultsOfQuery(
				"SELECT DISTINCT dataset FROM `dyad_dataset_approach_2_performance_samples` ORDER BY rand() LIMIT 10");

		while (resultSet.next()) {
			ResultSet actualResultSet = adapter.getResultsOfQuery(
					"SELECT * FROM `dyad_dataset_approach_2_performance_samples` WHERE dataset = ? ORDER BY score DESC LIMIT 2",
					Arrays.asList(resultSet.getString("dataset")));
			List<Vector> alternatives = new ArrayList<>();
			while (actualResultSet.next()) {
				DenseDoubleVector alternative = new DenseDoubleVector(
						Arrays.stream(actualResultSet.getString("pipeline_metafeatures").split(" "))
								.mapToDouble(Double::parseDouble).toArray());
				System.out.println(alternative.length());
				alternatives.add(alternative);
			}
			actualResultSet.beforeFirst();
			actualResultSet.next();
			SparseDyadRankingInstance instance = new SparseDyadRankingInstance(
					new DenseDoubleVector(Arrays.stream(actualResultSet.getString("dataset_metafeatures").split(" "))
							.mapToDouble(Double::valueOf).toArray()),
					alternatives);
			if (alternatives.size() > 1) {
				dataset.add(instance);
			}
		}
		return dataset;
	}

	public static void createTable(SQLAdapter adapter) throws Exception {
		ResultSet resultSet = adapter.getResultsOfQuery(
				"SELECT dataset, X_LANDMARKERS AS dataset_metafeatures, y AS pipeline_metafeatures, score FROM `dyad_dataset_approach_1_performance_samples` NATURAL JOIN dataset_metafeatures_mirror");

		ArrayList<Attribute> attInfo = new ArrayList<>();
		resultSet.next();
		Vector vector = new DenseDoubleVector(Arrays.stream(resultSet.getString("pipeline_metafeatures").split(" "))
				.mapToDouble(Double::parseDouble).toArray());
		for (int i = 0; i < vector.length(); i++) {
			attInfo.add(new Attribute("Att-"+i));
		}
		attInfo.add(new Attribute("Target"));
		Instances data = new Instances("Train", attInfo, 2000);
		data.setClassIndex(attInfo.size() - 1);
		
		//HashMap<Vector, Double> map = new HashMap<>();
		resultSet.beforeFirst();
		while (resultSet.next()) {
			//vector = new DenseDoubleVector(Arrays.stream(resultSet.getString("pipeline_metafeatures").split(" "))
					//.mapToDouble(Double::parseDouble).toArray());
			//map.put(vector, resultSet.getDouble("score"));
			double[] featureValues = Arrays.stream(resultSet.getString("pipeline_metafeatures").split(" "))
					.mapToDouble(Double::parseDouble).toArray();
			double[] attValues = new double[featureValues.length + 1];
			for (int i = 0; i < featureValues.length; i++) {
				attValues[i] = featureValues[i];
			}
			attValues[featureValues.length] = resultSet.getDouble("score");
			data.add(new DenseInstance(1, attValues));
		}
		

		
		RandomTreePerformanceBasedFeatureGenerator featureGen = new RandomTreePerformanceBasedFeatureGenerator();
		featureGen.disallowNonOccurence();
		featureGen.setNonOccurenceValue(-1);
		featureGen.setOccurenceValue(1);
		//featureGen.train(map);
		featureGen.train(data);
		System.out.println(featureGen);

//		resultSet.beforeFirst();
//		while (resultSet.next()) {
//			vector = new DenseDoubleVector(Arrays.stream(resultSet.getString("pipeline_metafeatures").split(" "))
//					.mapToDouble(Double::parseDouble).toArray());
//			HashMap<String, Object> insertionMap = new HashMap<String, Object>();
//			insertionMap.put("dataset", resultSet.getString("dataset"));
//			insertionMap.put("dataset_metafeatures", resultSet.getString("dataset_metafeatures"));
//			insertionMap.put("pipeline_metafeatures",
//					featureGen.predict(vector).stream().mapToObj(String::valueOf).collect(Collectors.joining(" ")));
//			insertionMap.put("score", resultSet.getDouble("score"));
//			// System.out.println(insertionMap);
//			adapter.insert("dyad_dataset_approach_2_performance_samples", insertionMap);
//		}
	}
}
