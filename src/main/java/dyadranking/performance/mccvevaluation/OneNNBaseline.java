package dyadranking.performance.mccvevaluation;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class OneNNBaseline {

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples_full";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final String metaFeatureName = "X_LANDMARKERS";

	private static Pattern arrayDeserializer = Pattern.compile(" ");

	private IBk ibk;

	private Instances trainDatasetIdInstances;

	private Map<String, List<DatabaseDyad>> cachedDyads;

	public OneNNBaseline(Map<String, List<DatabaseDyad>> cachedDyads) {
		this.cachedDyads = cachedDyads;
	}

	public void buildClassifier(List<Integer> trainDatasets, SQLAdapter adapter) throws Exception {
		List<Instance> datasetsAsInstance = new ArrayList<>();
		int classIndex = 0;
		ArrayList<Attribute> attributes = null;

		// Instances trainDatasetIdInstances;

		for (int trainDatasetId : trainDatasets) {

			// train 1-NN baseline
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

		trainDatasetIdInstances = new Instances("DatasetInstances", attributes, datasetsAsInstance.size());
		trainDatasetIdInstances.addAll(datasetsAsInstance);
		trainDatasetIdInstances.setClassIndex(classIndex);

		// 1NN
		ibk = new IBk(1);
		ibk.buildClassifier(trainDatasetIdInstances);
	}

	public double getClosestDatasetId(int datasetId, SQLAdapter adapter) throws Exception {
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
		return ibk.classifyInstance(instance);
	}

	private double[] getLandmarkersForDatasetId(int datasetId, SQLAdapter adapter) throws SQLException {
		ResultSet rs = adapter.getResultsOfQuery(
				"SELECT " + metaFeatureName + " FROM " + datasetMetaFeatureTable + " WHERE dataset = " + datasetId);
		rs.first();
		String serX = rs.getString(1);
		return arrayDeserializer.splitAsStream(serX).mapToDouble(Double::parseDouble).toArray();
	}

	public List<Pair<Double, Dyad>> get1NNRanking(double closestDatasetId, List<Dyad> list,
			String key){

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
	

}
