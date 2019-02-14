package dyadranking.zeroshotml.datagen;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
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
	
	private static final int SEED = 1;
	
	private static final String outputPath = "datasets/zeroshot/SMORBFtrain.dr";
	
	private static final int NUM_FEATURES = 3;
	
	private static final String PERF_SAMPLE_TABLE = "`smorbf_performance_samples`";
	
	private static final String DATASET_METAFEAT_TABLE = "`dataset_metafeatures_mirror`";
	
	private static final String J48_HYPERPARS = "C_pruning_confidence, M_min_inst";
	
	private static final String SMORBF_HYPERPARS = "C_complexity_const_exp, L_tolerance_exp, RBF_gamma_exp";
	
	private static final int[] DATASETS_TRAIN = { 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 28, 3, 30, 32 };
	
	private static final int[] DATASETS_TEST = { 5, 6, 36, 38, 44, 46 };
	
	private static final Pattern arrayDeserializer = Pattern.compile(" ");

	private static final double SAME_PERF_TOLERANCE = Math.pow(10,-5);

	private static final boolean EXCLUDE_SAME_PERF = true;

	private static final boolean TEST = false;
	
	public static List<Pair<double[], Double>> getSamplesForDataset(SQLAdapter adapter, int dataset) throws SQLException {
		ResultSet res = adapter.getResultsOfQuery(
				"SELECT " + SMORBF_HYPERPARS + ", performance "
				+ "FROM " + PERF_SAMPLE_TABLE
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
		
		System.out.println("Processing data set: " + dataset);
		
		return sampleList;	
	}
	
	public static double[] getDatasetLandmarkers(SQLAdapter adapter, int dataset) throws SQLException {
		ResultSet res = adapter.getResultsOfQuery(
				"SELECT X_LANDMARKERS FROM " + DATASET_METAFEAT_TABLE + " WHERE dataset = " + dataset );
		
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
		
		int[] datasets = DATASETS_TRAIN;
		if (TEST)
			datasets = DATASETS_TEST;

		
		for (int dataset : datasets) {
			Vector landmarkers = new DenseDoubleVector(getDatasetLandmarkers(adapter, dataset));
			List<Pair<double[], Double>> samples = getSamplesForDataset(adapter, dataset);
			int samplesLen = samples.size();
			for (int i = 0; i < samplesLen - 1; i++) {
				for (int j = i; j < samplesLen; j++) {
					List<Pair<double[], Double>> alternatives = new ArrayList<Pair<double[], Double>>(2);
					alternatives.add(samples.get(i));
					alternatives.add(samples.get(j));
					if(EXCLUDE_SAME_PERF 
					   && Math.abs(samples.get(i).getRight() - samples.get(j).getRight()) >= SAME_PERF_TOLERANCE) {
						Collections.sort(alternatives, new PerformanceComparator());
						data.add(new SparseDyadRankingInstance(landmarkers, 
							alternatives.stream()
							.map((Pair<double[], Double> p) -> new DenseDoubleVector(p.getLeft()))
							.collect(Collectors.toList())));
					}
				}
			}
		}
		
		Collections.shuffle(data, new Random(SEED));
		File outputFile = new File(outputPath);
		try {
			outputFile.createNewFile();
			BufferedOutputStream outputStream = new BufferedOutputStream(new FileOutputStream(outputFile));
			data.serialize(outputStream);
			outputStream.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}

}
