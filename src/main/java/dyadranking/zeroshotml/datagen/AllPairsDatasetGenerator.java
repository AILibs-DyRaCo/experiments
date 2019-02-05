package dyadranking.zeroshotml.datagen;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.nd4j.linalg.primitives.Pair;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.SparseDyadRankingInstance;

public class AllPairsDatasetGenerator {
	
	private static final String outputPath = "datasets/zeroshot/J48train.dr";
	
	private static final int NUM_FEATURES = 2;
	
	private static final String J48_HYPERPARS = "C_pruning_confidence, M_min_inst";
	
	private static final int[] datasets = { 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 28, 3, 30, 32 };
	
	//private static final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";
	
	private static final Pattern arrayDeserializer = Pattern.compile(" ");
	
	public static List<Pair<double[], Double>> getSamplesForDataset(SQLAdapter adapter, int dataset) throws SQLException {
		ResultSet res = adapter.getResultsOfQuery(
				"SELECT " + J48_HYPERPARS + ", performance "
				+ "FROM `j48_performance_samples`"
				+ "WHERE dataset = " + dataset);
		
		res.first();
		
		ArrayList<Pair<double[], Double>> sampleList = new ArrayList<Pair<double[], Double>>();
		
		do {
			double[] sample = new double[NUM_FEATURES];
			for (int i = 1; i <= NUM_FEATURES; i++)
				sample[i - 1] = res.getDouble(i);
			Pair<double[], Double> samplePerfPair = new Pair<double[], Double>(sample, res.getDouble(NUM_FEATURES + 1));
			sampleList.add(samplePerfPair);
		} while (res.next());
		
		return sampleList;	
	}
	
	public static double[] getDatasetLandmarkers(SQLAdapter adapter, int dataset) throws SQLException {
		ResultSet res = adapter.getResultsOfQuery(
				"SELECT X_LANDMARKERS FROM dataset_metafeatures_mirror WHERE dataset = " + dataset );
		
		res.first();	
		String serializedY = res.getString(1);
		double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble).toArray();
		
		return yArray;
	}
	
	private static class PerformanceComparator implements Comparator<Pair<double[], Double>> {

		@Override
		public int compare(Pair<double[], Double> o1, Pair<double[], Double> o2) {
			return Double.compare(- o1.getRight(), - o2.getRight());
		}
		
	}
	
	public static void main(String[] args) throws SQLException {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		
		DyadRankingDataset data = new DyadRankingDataset();
		
		for (int dataset : datasets) {
			Vector landmarkers = new DenseDoubleVector(getDatasetLandmarkers(adapter, dataset));
			List<Pair<double[], Double>> samples = getSamplesForDataset(adapter, dataset);
			int samplesLen = samples.size();
			for (int i = 0; i < samplesLen - 1; i++) {
				for (int j = 1; j < samplesLen; j++) {
					List<Pair<double[], Double>> alternatives = new ArrayList<Pair<double[], Double>>(2);
					alternatives.add(samples.get(i));
					alternatives.add(samples.get(j));
					Collections.sort(alternatives, new PerformanceComparator());
					data.add(new SparseDyadRankingInstance(landmarkers, 
							alternatives.stream()
							.map((Pair<double[], Double> p) -> new DenseDoubleVector(p.getLeft()))
							.collect(Collectors.toList())));
				}
			}
		}
		
		Collections.shuffle(data);
		
		File outputFile = new File(outputPath);
		try {
			outputFile.createNewFile();
			data.serialize(new FileOutputStream(outputFile));
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

}
