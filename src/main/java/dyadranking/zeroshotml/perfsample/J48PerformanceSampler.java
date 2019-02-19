package dyadranking.zeroshotml.perfsample;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.aeonbits.owner.ConfigCache;

import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Generates performance samples in terms of the relative 0/1 loss for a J48 classifier
 * on stratified 0.8/0.2 splits from a grid search over the numerical hyper-parameters.
 * 
 * C: pruning confidence
 * M: minimum number of instances per leaf
 * 
 * @author Michael Braun
 *
 */
public class J48PerformanceSampler {
	
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
		IJ48PerformanceSamplerConfig m = ConfigCache.getOrCreate(IJ48PerformanceSamplerConfig.class);
		if (m.getDatasetFolder() == null || !m.getDatasetFolder().exists())
			throw new IllegalArgumentException("config specifies invalid dataset folder " + m.getDatasetFolder());

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
				String C_pruning_confidence_option = "-C " + description.get("C_pruning_confidence");
				String M_min_inst_option = " -M " + description.get("M_min_inst");

				J48 j48 = new J48();
				String options = 
						C_pruning_confidence_option +
						M_min_inst_option;
				String[] optionsSplit = Utils.splitOptions(options);
				j48.setOptions(optionsSplit);
				
				/* train and evaluate */
				j48.buildClassifier(dataTrain);
				Evaluation eval = new Evaluation(dataTest);
				eval.evaluateModel(j48, dataTest);
				double performance = eval.pctCorrect();

				Map<String, Object> results = new HashMap<>();
				results.put("performance", performance);
				processor.processResults(results);
			}
		});
		runner.randomlyConductExperiments(true);
	}
}
