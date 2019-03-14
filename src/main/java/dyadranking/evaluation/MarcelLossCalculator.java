package dyadranking.evaluation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

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

	private static final int K = 10;

	private static final String filePath = "./predictions/prediction-random-MLPlan-Data.txt-18-50.txt";

	private static final String scalerPath = "./scalers/scaler-random-MLPlan-Data.txt-18-.ser";

	private static String dyadTable = "dyad_dataset_approach_5_performance_samples_full";

	private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";

	private static final String metaFeatureName = "X_LANDMARKERS";

	private static Pattern arrayDeserializer = Pattern.compile(" ");

	private static Map<Dyad, Double> cachedDyads = new HashMap<Dyad, Double>();

	private static boolean transformAlternatives = true;

	public static void main(String args[]) {
		
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		DyadRankingDataset currentPredictions = new DyadRankingDataset();

		// deserialize the scaler used for these predictions
		AbstractDyadScaler scaler = null;
		FileInputStream fis;
		try {
			fis = new FileInputStream(scalerPath);
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

		try {
			currentPredictions.deserialize(new FileInputStream(new File(filePath)));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		if(scaler instanceof DyadMinMaxScaler) {
			DyadMinMaxScaler mmscaler = (DyadMinMaxScaler) scaler;
			if(transformAlternatives)
				((DyadMinMaxScaler) scaler).untransformAlternatives(currentPredictions);
		}
		
		for (IInstance instance : currentPredictions) {

			// get the String representation that is consistent with the database entry
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			Vector instanceFeatureVector = ((DenseDoubleVector) drInstance.getDyadAtPosition(0).getInstance());
			String instanceFeatures = instanceFeatureVector.toString();
//			System.out.println(instanceFeatures);
			String databaseMetafeatures = instanceFeatures.replaceAll(",", " ");
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

				// extract alternative features
//				System.out.println(alternativeFeatures);
				List<Pair<Dyad, Double>> trueDyadLosses = new ArrayList<Pair<Dyad, Double>>();
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
				System.out.println(allAlternatives);
				resultsPipelines = adapter.getResultsOfQuery("SELECT pipeline_id, y, score FROM " + dyadTable
						+ " WHERE dataset = " + datasetID + " AND y IN " + allAlternatives);
				while (resultsPipelines.next()) {
					double loss = resultsPipelines.getDouble(3);
					String serializedY = resultsPipelines.getString(2);
//					System.out.println(datasetID);
//					System.out.println(resultsPipelines.getInt(1));
//					System.out.println("loss: " + loss);
//					System.out.println();

					double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble)
							.toArray();
					Dyad dyad = new Dyad(instanceFeatureVector, new DenseDoubleVector(yArray));
//					if (transformAlternatives) {
//						if(scaler == null)
//							throw new IllegalStateException("No dyad scaler found!");
//						scaler.transformAlternatives(dyad, new ArrayList<Integer>());
//					}
					System.out.println(dyad);
					cachedDyads.put(dyad, loss);
				}

			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			for (IInstance inst : currentPredictions) {
				IDyadRankingInstance drPredictedInstance = (IDyadRankingInstance) inst;
				System.out.println("\nNew ranking");
				for (Dyad dyad : drPredictedInstance) {
					if(!cachedDyads.containsKey(dyad))
						System.out.println(dyad.getAlternative());
					
				}
			}
		}

//		List<Pair<Double, Dyad>> testDyads = getDyadRankingInstanceForDataset(datasetId, adapter);
//
//		List<Dyad> orderedDyads = testDyads.stream().sorted(DyadComparator::compare).map(Pair::getY)
//				.collect(Collectors.toList());
//		
//		Map<Object, Object> map = testDyads.stream().collect(Collectors.toMap(Pair::getY, Pair::getX));

	}
}
