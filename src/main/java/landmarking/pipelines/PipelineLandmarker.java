package landmarking.pipelines;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.aeonbits.owner.ConfigCache;

import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WekaMLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WekaMLPlanWekaUtil;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import jaicore.basic.SQLAdapter;
import jaicore.experiments.ExperimentDBEntry;
import jaicore.experiments.ExperimentRunner;
import jaicore.experiments.IExperimentIntermediateResultProcessor;
import jaicore.experiments.IExperimentSetConfig;
import jaicore.experiments.IExperimentSetEvaluator;
import jaicore.ml.core.evaluation.measure.singlelabel.ZeroOneLoss;
import jaicore.ml.evaluation.evaluators.weka.MonteCarloCrossValidationEvaluator;
import jaicore.ml.evaluation.evaluators.weka.SimpleEvaluatorMeasureBridge;
import jaicore.ml.openml.OpenMLHelper;
import jaicore.search.core.interfaces.GraphGenerator;
import jaicore.search.structure.graphgenerator.SingleRootGenerator;
import weka.core.Instances;

public class PipelineLandmarker {

	public static void main(final String[] args) {
		IPipelineLandmarkingConfig m = ConfigCache.getOrCreate(IPipelineLandmarkingConfig.class);
		ExperimentRunner runner = new ExperimentRunner(new IExperimentSetEvaluator() {

			@Override
			public IExperimentSetConfig getConfig() {
				return m;
			}

			@Override
			public void evaluate(final ExperimentDBEntry experimentEntry, final SQLAdapter adapter,
					final IExperimentIntermediateResultProcessor processor) throws Exception {

				/* get experiment setup */
				Map<String, String> description = experimentEntry.getExperiment().getValuesOfKeyFields();
				String datasetID = description.get("dataset_id");
				String openMLKey = description.get("openMLKey");
				int mccvRepeats = Integer.parseInt(description.get("mccvRepeats"));
				int mccvSeed = Integer.parseInt(description.get("mccvSeed"));
				double mccvTrainRatio = Double.parseDouble(description.get("mccvTrainRatio"));

				OpenMLHelper.setApiKey(openMLKey);
				Instances data = OpenMLHelper.getInstancesById(Integer.parseInt(datasetID));
				data.setClassIndex(data.numAttributes() - 1);
				
				MLPlanWekaClassifier mlplan = new WekaMLPlanWekaClassifier();
				mlplan.setData(data);
				
				GraphGenerator gg = mlplan.getGraphGenerator();
				SingleRootGenerator srg = (SingleRootGenerator<T>)
				
				List<MLPipeline> allPipelines = WekaMLPlanWekaUtil.getAllLegalWekaPipelinesWithDefaultConfig();

				MonteCarloCrossValidationEvaluator evaluator = new MonteCarloCrossValidationEvaluator(
						new SimpleEvaluatorMeasureBridge(new ZeroOneLoss()), mccvRepeats, data, mccvTrainRatio,
						mccvSeed);
				for (MLPipeline pipeline : allPipelines) {
					double loss = evaluator.evaluate(pipeline);
				}

				/* run experiment */
				Map<String, Object> results = new HashMap<>();

				/* report results */
//				results.put("loss", loss);
//				processor.processResults(results);
			}
		});
//		runner.randomlyConductExperiments(true);
	}

}
