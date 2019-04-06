package dyadranking.nodeevaluation.timebased;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.sql.Time;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.aeonbits.owner.ConfigFactory;

import de.upb.crc901.mlplan.core.MLPlan;
import de.upb.crc901.mlplan.core.MLPlanBuilder;
import de.upb.crc901.mlplan.metamining.dyadranking.DyadRankingBasedNodeEvaluator;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.serialization.ComponentLoader;
import jaicore.basic.SQLAdapter;
import jaicore.basic.TimeOut;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.WekaUtil;
import jaicore.ml.cache.ReproducibleInstances;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class MLPlanExperimenter implements IExperimentSetEvaluator {

	private MLPlanTimeBasedExperimenterConfig config;

	public MLPlanExperimenter(File configFile) {
		Properties props = new Properties();
		try {
			props.load(new FileInputStream(configFile));
		} catch (IOException e) {
			System.err.println("Could not find or access config file " + configFile);
			System.exit(1);
		}
		this.config = ConfigFactory.create(MLPlanTimeBasedExperimenterConfig.class, props);
	}

	@Override
	public IExperimentSetConfig getConfig() {
		return config;
	}

	@Override
	public void evaluate(ExperimentDBEntry experimentEntry, SQLAdapter adapter,
			IExperimentIntermediateResultProcessor processor) throws Exception {
		// Put ArrayjobNr in fields to identify error logs from db table
		Map<String, Object> results = new HashMap<>();

		Map<String, String> keyfields = experimentEntry.getExperiment().getValuesOfKeyFields();
		System.out.println("Start experiment with keys: " + keyfields);

		boolean useDyadRanking = new Boolean(keyfields.get("useDyadRanking"));
		String dataset = keyfields.get("dataset");
		int seed = Integer.parseInt(keyfields.get("seed"));
		long timeout = Long.parseLong(keyfields.get("timeout"));


		ReproducibleInstances data = ReproducibleInstances.fromOpenML(dataset, "4350e421cdc16404033ef1812ea38c01");
		data.setClassIndex(data.numAttributes() - 1);
		List<Instances> split = WekaUtil.getStratifiedSplit((Instances) data, (new Random(0)).nextLong(), 0.7d);

		MLPlanBuilder builder = new MLPlanBuilder();

		builder.withSearchSpaceConfigFile(new File("conf/automl/searchmodels/weka/weka-approach-5-autoweka.json"));
		builder.withRandomCompletionBasedBestFirstSearch();

		if (useDyadRanking) {
			builder = builder.withAutoWEKAConfiguration(false);
			builder.withPreferredNodeEvaluator(new DyadRankingBasedNodeEvaluator<>(
					new ComponentLoader(new File("conf/automl/searchmodels/weka/weka-approach-5-autoweka.json"))));
		} else {
			builder = builder.withAutoWEKAConfiguration();
		}

		builder.withTimeoutForNodeEvaluation(new TimeOut(60, TimeUnit.SECONDS));
		builder.withTimeoutForSingleSolutionEvaluation(new TimeOut(30, TimeUnit.SECONDS));

		MLPlan mlplan = new MLPlan(builder, split.get(0));

		mlplan.setPortionOfDataForPhase2(0.3f);
		mlplan.setRandomSeed(seed);
		mlplan.setLoggerName("mlplan");
		mlplan.setTimeout(timeout, TimeUnit.SECONDS);
		mlplan.setNumCPUs(1);
		// mlplan.registerListener(logger);
		System.out.println("Starting");
		try {
			Classifier optimizedClassifier = mlplan.call();
			System.out.println("Finished build of the classifier.");
			System.out.println("Chosen model is: " + ((MLPipeline) mlplan.getSelectedClassifier()).toString());

			/* evaluate solution produced by mlplan */
			Evaluation eval = new Evaluation(split.get(0));
			eval.evaluateModel(optimizedClassifier, split.get(1));

			System.out.println("Error is " + eval.errorRate());
			results.put("loss", Double.toString(eval.errorRate()));
			processor.processResults(results);
			processor.processResults(results);

		} catch (Exception e) {
			System.err.println("amk mlplan");
			e.printStackTrace();
		}

	}

	private static void print(final String message) {
		System.out.println(new Time(System.currentTimeMillis()).toString() + ": " + message);
	}

	public static void main(final String[] args) {
		/* check config */
		print("Start experiment runner...");

		print("Conduct random experiment...");
		for (int i = 0; i < 4; i++) {
			new Thread(() -> {
				ExperimentRunner runner = new ExperimentRunner(
						new MLPlanExperimenter(new File("conf/timebasedevals/timebased.properties")));
				runner.randomlyConductExperiments(-1, true);
				print("Experiment conducted.");
			}).start();
		}
	}

}
