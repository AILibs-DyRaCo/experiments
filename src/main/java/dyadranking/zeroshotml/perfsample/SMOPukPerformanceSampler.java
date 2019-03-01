package dyadranking.zeroshotml.perfsample;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.aeonbits.owner.ConfigCache;

import dyadranking.zeroshotml.perfsample.classifiersuppliers.IClassifierSupplier;
import dyadranking.zeroshotml.perfsample.classifiersuppliers.SMOPKSupplier;
import dyadranking.zeroshotml.perfsample.classifiersuppliers.SMOpukSupplier;
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

public class SMOPukPerformanceSampler {

	public static void main(String[] args) {
		ISMOPKPerformanceSamplerConfig m = ConfigCache.getOrCreate(ISMOPKPerformanceSamplerConfig.class);
		if (m.getDatasetFolder() == null || !m.getDatasetFolder().exists())
			throw new IllegalArgumentException("config specifies invalid dataset folder " + m.getDatasetFolder());
		
		IClassifierSupplier clsSup = new SMOpukSupplier();
		
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
								+ dataset))));
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
