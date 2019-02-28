package dyadranking.zeroshotml.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.aeonbits.owner.ConfigCache;
import org.aeonbits.owner.ConfigFactory;

import dyadranking.zeroshotml.perfsample.IJ48PerformanceSamplerConfig;
import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import jaicore.ml.dyadranking.zeroshot.util.ZeroShotUtil;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;

public class ZeroShotCurvesEvaluator {
	
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
		put("183", "dataset_183_abalone.arff");
	}};
	
	private static J48 setupJ48(Double[] params) throws Exception {
		J48 j48 = new J48();
		j48.setOptions(ZeroShotUtil.mapJ48InputsToWekaOptions(params[0], params[1]));
				
		return j48;
	}
	
	private static RandomForest setupRF(Double[] params, int numAttributes) throws Exception {
		RandomForest rf = new RandomForest();
		rf.setOptions(ZeroShotUtil.mapRFInputsToWekaOptions(params[0], params[1], params[2], params[3], numAttributes));
		
		return rf;
	}
	
	private static SMO setupSMO(Double[] params) throws Exception {
		SMO smo = new SMO();
		smo.setOptions(ZeroShotUtil.mapSMORBFInputsToWekaOptions(params[0], params[1], params[2]));
		
		return smo;
	}
	
	private static MultilayerPerceptron setupMLP(Double[] params) throws Exception {
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		mlp.setOptions(ZeroShotUtil.mapMLPInputsToWekaOptions(params[0], params[1], params[2]));
		
		return mlp;
	}
	
	public static void main(String[] args) {
		IZeroShotCurvesEvaluatorConfig m = ConfigCache.getOrCreate(IZeroShotCurvesEvaluatorConfig.class);
		if (m.getDatasetFolder() == null || !m.getDatasetFolder().exists())
			throw new IllegalArgumentException("config specifies invalid dataset folder " + m.getDatasetFolder());
		
		IAuxZeroShotCurvesEvaluatorConfig aux = ConfigFactory.create(IAuxZeroShotCurvesEvaluatorConfig.class);		
		
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
				
				String paramsFileName = description.get("classifier") + description.get("paramsfile_prefix") + description.get("data_set")	+ description.get("paramsfile_suffix");
				List<Double[]> paramsList = new ArrayList<Double[]>();
				try (BufferedReader reader = new BufferedReader(
							new FileReader(aux.getParamsFolder()
									+ File.separator
									+ paramsFileName)))
				{
					String line;
					while ((line = reader.readLine()) != null) {
						String[] params = line.split(",");
						paramsList.add(Arrays.stream(params).map(Double::valueOf).toArray(Double[]::new));
					}
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				Double[] params = paramsList.get(Integer.parseInt(description.get("iteration_index")));
				
				Instances evalData = new Instances(new BufferedReader(
						new FileReader(new File(m.getDatasetFolder() 
								+ File.separator 
								+ datasetIdMap.get(description.get("data_set"))))));
				evalData.setClassIndex(evalData.numAttributes() - 1);
				Evaluation eval = new Evaluation(evalData);
				double score = 0.0d;
				Classifier classifier = null;
				switch(description.get("classifier")) {
				case "j48":
					classifier = setupJ48(params);
					break;
				case "rf":
					classifier = setupRF(params, evalData.numAttributes());
					break;
				case "smo":
					classifier = setupSMO(params);
					break;
				case "mlp":
					classifier = setupMLP(params);
					break;
				}				
				eval.crossValidateModel(classifier, evalData, 5, new Random(Integer.parseInt(description.get("cv_seed"))));
				score = eval.pctCorrect();
				
				StringBuilder paramsString = new StringBuilder();
				for(int i = 0; i < params.length - 1; i++) {
					paramsString.append(params[i]).append(", ");
				}
				paramsString.append(params[params.length-1]);
				
				Map<String, Object> results = new HashMap<>();
				results.put("parameter_file", paramsFileName);
				results.put("percent_correct", score);
				results.put("parameters", paramsString);
				processor.processResults(results);
			}
		});
		runner.randomlyConductExperiments(true);
	}
}
