package dyadranking.evaluation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import javax.management.InstanceAlreadyExistsException;

import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.nd4j.linalg.primitives.Pair;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.util.AbstractDyadScaler;
import jaicore.ml.dyadranking.util.DyadMinMaxScaler;

public class MarcelLossCalculator {

	private static final String filePathEx = "./predictions/prediction-random-MLPlan-Data.txt-18-50.txt";

	private static final String scalerPathEx = "./scalers/scaler-random-MLPlan-Data.txt-18-.ser";

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples_full";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final String metaFeatureName = "X_LANDMARKERS";

	private static Pattern arrayDeserializer = Pattern.compile(" ");

	private static Map<Dyad, Double> cachedDyads = new HashMap<Dyad, Double>();

	private static final boolean transformAlternativesEx = true;

	public static void main(String args[]) {

		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		DyadRankingDataset dataset = new DyadRankingDataset();
		AbstractDyadScaler scaler = null;
		try {
			dataset.deserialize(new FileInputStream(new File(filePathEx)));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			FileInputStream fis;
			fis = new FileInputStream(scalerPathEx);
			ObjectInputStream ois = new ObjectInputStream(fis);
			scaler = (AbstractDyadScaler) ois.readObject();
			ois.close();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		loadDyadScores(adapter, dataset, scaler, transformAlternativesEx);
		computeMarcelLossAvg(3, dataset.get(0));
		computeMarcelLossMin(3, dataset.get(0));
		computeMarcelLossAvg(5, dataset.get(0));
		computeMarcelLossMin(5, dataset.get(0));
		computeMarcelLossAvg(10, dataset.get(0));
		computeMarcelLossMin(10, dataset.get(0));
		
	}

	public static void computeMarcelLossMin(int k, IDyadRankingInstance predictedRanking) {
		DescriptiveStatistics predictedStats = new DescriptiveStatistics();
		DescriptiveStatistics trueStats = new DescriptiveStatistics();

		List<Pair<Dyad, Double>> dyadsAndScores = new ArrayList<Pair<Dyad, Double>>(predictedRanking.length());
		for (Dyad dyad : predictedRanking) {
			dyadsAndScores.add(new Pair<Dyad, Double>(dyad, cachedDyads.get(dyad)));
		}
		Collections.sort(dyadsAndScores, Comparator.comparing(p -> p.getRight()));
		for (int i = 0; i < k; i++) {
			System.out.println(i + "th predicted loss: " + cachedDyads.get(predictedRanking.getDyadAtPosition(i))
					+ "\t\t true loss: " + dyadsAndScores.get(i).getRight());
			predictedStats.addValue(cachedDyads.get(predictedRanking.getDyadAtPosition(i)));
			trueStats.addValue(dyadsAndScores.get(i).getRight());
		}
		System.out.println("predicted min: " + predictedStats.getMin());
		System.out.println("true min: " + trueStats.getMin());
		double weverNumber = Math.abs(predictedStats.getMin() - trueStats.getMin());
		System.out.println("wever number min: " + weverNumber);
	}

	public static void computeMarcelLossAvg(int k, IDyadRankingInstance predictedRanking) {
		DescriptiveStatistics predictedStats = new DescriptiveStatistics();
		DescriptiveStatistics trueStats = new DescriptiveStatistics();

		List<Pair<Dyad, Double>> dyadsAndScores = new ArrayList<Pair<Dyad, Double>>(predictedRanking.length());
		for (Dyad dyad : predictedRanking) {
			dyadsAndScores.add(new Pair<Dyad, Double>(dyad, cachedDyads.get(dyad)));
		}
		Collections.sort(dyadsAndScores, Comparator.comparing(p -> p.getRight()));
		for (int i = 0; i < k; i++) {
			System.out.println(i + "th predicted loss: " + cachedDyads.get(predictedRanking.getDyadAtPosition(i))
					+ "\t\t true loss: " + dyadsAndScores.get(i).getRight());
			predictedStats.addValue(cachedDyads.get(predictedRanking.getDyadAtPosition(i)));
			trueStats.addValue(dyadsAndScores.get(i).getRight());
		}
		System.out.println("predicted avg: " + predictedStats.getMean());
		System.out.println("true avg: " + trueStats.getMean());
		double weverNumber = Math.abs(predictedStats.getMean() - trueStats.getMean());
		System.out.println("wever number avg: " + weverNumber);
	}

	public static void loadDyadScores(SQLAdapter adapter, DyadRankingDataset currentPredictions, AbstractDyadScaler scaler,
			boolean transformAlternatives) {

		if (scaler instanceof DyadMinMaxScaler) {
			DyadMinMaxScaler mmscaler = (DyadMinMaxScaler) scaler;
			if (transformAlternatives)
				((DyadMinMaxScaler) scaler).untransformAlternatives(currentPredictions, 3);
		}

		for (IInstance instance : currentPredictions) {

			// get the String representation that is consistent with the database entry
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			Vector instanceFeatureVector = ((DenseDoubleVector) drInstance.getDyadAtPosition(0).getInstance());
			String instanceFeatures = instanceFeatureVector.toString();
//			System.out.println(instanceFeatures);
			String databaseMetafeatures = instanceFeatures.replaceAll(",", " ");
			databaseMetafeatures = databaseMetafeatures.replaceAll(" 0.0 ", " 0 ");
			databaseMetafeatures = databaseMetafeatures.substring(1, databaseMetafeatures.length() - 1);
//			System.out.println(databaseMetafeatures);

			try {
				ResultSet resultsDataset;
				resultsDataset = adapter.getResultsOfQuery("SELECT dataset FROM " + datasetMetaFeatureTable + " WHERE "
						+ metaFeatureName + " = \'" + databaseMetafeatures + "\'");
				int datasetID = -1;
				while (resultsDataset.next()) {
					datasetID = resultsDataset.getInt(1);
//					System.out.println(datasetID);
				}

				if (datasetID == -1) {
					System.out.println(databaseMetafeatures);
				}

				// extract alternative features
//				System.out.println(alternativeFeatures);
//				List<Pair<Dyad, Double>> trueDyadLosses = new ArrayList<Pair<Dyad, Double>>();
				ResultSet resultsPipelines;
				StringBuilder sb = new StringBuilder();
				sb.append("(");
				String prefix = "";
				for (Dyad dyad : drInstance) {
					// create a String for the SQL query

					String alternativeFeatureString = dyad.getAlternative().toString();
					alternativeFeatureString = alternativeFeatureString.replaceAll(",", " ");
					alternativeFeatureString = alternativeFeatureString.substring(1,
							alternativeFeatureString.length() - 1);
					sb.append(prefix);
					prefix = ", ";
					sb.append("\'");
					sb.append(alternativeFeatureString);
					sb.append("\'");
				}
				sb.append(")");
				String allAlternatives = sb.toString();
				resultsPipelines = adapter.getResultsOfQuery("SELECT pipeline_id, y, score FROM " + dyadTable
						+ " WHERE dataset = " + datasetID + " AND y IN " + allAlternatives);
				boolean isEmpty = true;
				while (resultsPipelines.next()) {
					double loss = resultsPipelines.getDouble(3);
					String serializedY = resultsPipelines.getString(2);
//						System.out.println(datasetID);
//						System.out.println(resultsPipelines.getInt(1));
//						System.out.println("loss: " + loss);
//						System.out.println();

					double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble)
							.toArray();
					Dyad dyad = new Dyad(instanceFeatureVector, new DenseDoubleVector(yArray));
//					if (transformAlternatives) {
//						if(scaler == null)
//							throw new IllegalStateException("No dyad scaler found!");
//						scaler.transformAlternatives(dyad, new ArrayList<Integer>());
//					}
//					System.out.println(dyad);
//					System.out.println(loss);
					cachedDyads.put(dyad, loss);
					isEmpty = false;
				}
//				if (isEmpty) {
//					System.out.println(
//							"RESULTSET IS EMPTY FOR DATASET " + datasetID + " PIPELINE " + databaseAlternative);
//				}

			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

}
