package landmarking.pipelines;

import java.io.File;
import java.nio.file.Paths;
import java.sql.ResultSet;
import java.util.HashMap;
import java.util.Map;

import org.aeonbits.owner.ConfigCache;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.model.ComponentInstance;
import hasco.serialization.ComponentLoader;
import hasco.serialization.HASCOJacksonModule;
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
				String pipelineID = description.get("pipeline_id");
				String openMLKey = description.get("openMLKey");
				int mccvRepeats = Integer.parseInt(description.get("mccvRepeats"));
				int mccvSeed = Integer.parseInt(description.get("mccvSeed"));
				double mccvTrainRatio = Double.parseDouble(description.get("mccvTrainRatio"));

				OpenMLHelper.setApiKey(openMLKey);
				Instances data = OpenMLHelper.getInstancesById(Integer.parseInt(datasetID));
				data.setClassIndex(data.numAttributes() - 1);
				String compositionString = "";
				ResultSet rs = adapter
						.getResultsOfQuery("SELECT composition FROM draco_pipelines WHERE pipeline_id = " + pipelineID);
				rs.first();
				compositionString = rs.getString(1);

				File jsonFile = Paths.get(getClass().getClassLoader()
						.getResource(Paths.get("automl", "searchmodels", "weka", "weka-all-autoweka.json").toString())
						.toURI()).toFile();

				ComponentLoader loader = new ComponentLoader(jsonFile);
				ObjectMapper mapper = new ObjectMapper();
				mapper.registerModule(new HASCOJacksonModule(loader.getComponents()));

				ComponentInstance composition = mapper.readValue(compositionString, ComponentInstance.class);
				System.out.println(composition);
				WEKAPipelineFactory factory = new WEKAPipelineFactory();
				MLPipeline pipeline = factory.getComponentInstantiation(composition);
				MonteCarloCrossValidationEvaluator evaluator = new MonteCarloCrossValidationEvaluator(
						new SimpleEvaluatorMeasureBridge(new ZeroOneLoss()), mccvRepeats, data, mccvTrainRatio,
						mccvSeed);
				double loss = evaluator.evaluate(pipeline);
				/* run experiment */
				Map<String, Object> results = new HashMap<>();

				/* report results */
				results.put("loss", loss);
				processor.processResults(results);
			}
		});
		runner.randomlyConductExperiments(true);
	}

}
