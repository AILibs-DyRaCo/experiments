package dyadranking.zeroshotml.perfsample;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.aeonbits.owner.ConfigCache;

import dyadranking.zeroshotml.perfsample.classifiersuppliers.IBkSupplier;
import dyadranking.zeroshotml.perfsample.classifiersuppliers.IClassifierSupplier;
import dyadranking.zeroshotml.perfsample.classifiersuppliers.LogisticSupplier;
import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class LogisticPerformanceSampler {

		private static Map<String, String> datasetIdMap = new HashMap<String, String>() {{
			put("12", "dataset_12_mfeat-factors.arff");
			put("14", "dataset_14_mfeat-fourier.arff");
			put("16", "dataset_16_mfeat-karhunen.arff");
			put("18", "dataset_18_mfeat-morphological.arff");
			put("20", "dataset_20_mfeat-pixel.arff");
			put("21", "dataset_21_car.arff");
			put("22", "dataset_22_mfeat-zernike.arff");
			put("23", "dataset_23_cmc.arff");
			put("24", "dataset_24_mushroom.arff");
			put("26", "dataset_26_nursery.arff");
			put("28", "dataset_28_optdigits.arff");
			put("3", "dataset_3_kr-vs-kp.arff");
			put("30", "dataset_30_page-blocks.arff");
			put("32", "dataset_32_pendigits.arff");
			put("5", "dataset_5_arrhythmia.arff");
			put("6", "dataset_6_letter.arff");
			put("36", "dataset_36_segment.arff");
			put("38", "dataset_38_sick.arff");
			put("44", "dataset_44_spambase.arff");
			put("46", "dataset_46_splice.arff");
	}};
	
	public static void main(String[] args) {
		ILogisticPerformanceSamplerConfig m = ConfigCache.getOrCreate(ILogisticPerformanceSamplerConfig.class);
		if (m.getDatasetFolder() == null || !m.getDatasetFolder().exists())
			throw new IllegalArgumentException("config specifies invalid dataset folder " + m.getDatasetFolder());
		
		IClassifierSupplier clsSup = new LogisticSupplier();
		
		ExperimentRunner runner = new ExperimentRunner(new IExperimentSetEvaluator() {
	
			@Override
			public IExperimentSetConfig getConfig() {
				return m;
			}
	
			@Override
			public void evaluate(ExperimentDBEntry experimentEntry, SQLAdapter adapter,
					IExperimentIntermediateResultProcessor processor) throws Exception {
				
				/* get experiment setup */
				Map<String, String> description = experimentEntry.getExperiment().getValuesOfKeyFields();
				
				/* set up data */
				String dataset = description.get("dataset");
				Instances data = new Instances(new BufferedReader(
						new FileReader(new File(m.getDatasetFolder() 
								+ File.separator 
								+ datasetIdMap.get(dataset)))));
				data.setClassIndex(data.numAttributes() - 1);
				
				List<Instances> dataSplit = WekaUtil.getStratifiedSplit(data, 0, 0.8, 0.2);
				Instances dataTrain = dataSplit.get(0);
				Instances dataTest = dataSplit.get(1);
				
				/* set up classifier */
				Classifier cls = clsSup.setupClassifier(description, data);
				
				/* train and evaluate */
				cls.buildClassifier(dataTrain);
				Evaluation eval = new Evaluation(dataTest);
				eval.evaluateModel(cls, dataTest);
				double performance = eval.pctCorrect();
	
				Map<String, Object> results = new HashMap<>();
				results.put("performance", performance);
				processor.processResults(results);
			}
		});
		runner.randomlyConductExperiments(true);
	}

}
