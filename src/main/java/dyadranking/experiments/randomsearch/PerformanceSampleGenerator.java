package dyadranking.experiments.randomsearch;

import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import org.aeonbits.owner.ConfigFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.core.Util;
import hasco.model.ComponentInstance;
import jaicore.basic.SQLAdapter;
import jaicore.ml.cache.ReproducibleInstances;
import jaicore.ml.core.evaluation.measure.singlelabel.ZeroOneLoss;
import jaicore.ml.evaluation.evaluators.weka.MonteCarloCrossValidationEvaluator;
import jaicore.ml.evaluation.evaluators.weka.SimpleEvaluatorMeasureBridge;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.algorithms.standard.random.RandomSearch;
import jaicore.search.core.interfaces.GraphGenerator;
import jaicore.search.model.other.SearchGraphPath;
import jaicore.search.model.probleminputs.GraphSearchInput;

/**
 * Simple class that can be used to generate performance samples from the
 * ML-Plan search space for datasets from OpenML. Therefore a
 * {@link RandomSearch} is applied to the search space graph return by ML-Plan.
 * 
 * @author Jonas Hanselle
 *
 */
public class PerformanceSampleGenerator {

	public static void main(String[] args) {
		String openmlID = args[0];
		PerformanceSampleGenerator psg = new PerformanceSampleGenerator();
		psg.evaluate(openmlID);
	}

	private IPerformanceSampleGeneratorConfig config;

	public PerformanceSampleGenerator() {
		this.config = ConfigFactory.create(IPerformanceSampleGeneratorConfig.class);
	}

	public void evaluate(String openmlID) {

		try {
			SQLAdapter adapter = new SQLAdapter(config.getDBHost(), config.getDBUsername(), config.getDBPassword(),
					config.getDBDatabaseName());
			
			ZeroOneLoss lossFunction = new ZeroOneLoss();
			ObjectMapper mapper = new ObjectMapper();
			
			//TODO jonas: insert components here
			MLPlanWekaClassifier mlplan = new MLPlanWekaMLPlanWekaClassifier();
			ReproducibleInstances data = ReproducibleInstances.fromOpenML(openmlID, config.getOpenMLKey());

			data.setClassIndex(data.numAttributes() - 1);
			mlplan.setLoggerName("mlplan");
			mlplan.setData(data);
			
			GraphGenerator gg = mlplan.getGraphGenerator();
			GraphSearchInput gsi = new GraphSearchInput(gg);
			RandomSearch rs = new RandomSearch(gsi, config.getRandomSearchSeed());
			while (rs.hasNext()) {
				SearchGraphPath sgp = rs.nextSolution();
				TFDNode goalNode = (TFDNode) sgp.getNodes().get(sgp.getNodes().size() - 1);
				WEKAPipelineFactory factory = new WEKAPipelineFactory();
				ComponentInstance ci = Util.getSolutionCompositionFromState(mlplan.getComponents(), goalNode.getState(),
						true);
				try {
					MLPipeline mlp = factory.getComponentInstantiation(ci);
					MonteCarloCrossValidationEvaluator evaluator = new MonteCarloCrossValidationEvaluator(
							new SimpleEvaluatorMeasureBridge(lossFunction), config.getNumberMCCVRepeats(), data,
							config.getMCCVTrainRatio(), config.getMCCVSeed());
					double score = evaluator.evaluate(mlp);
					
					
					//insert into DB
					Map<String, String> valueMap = new HashMap<>();
					valueMap.put("composition", mapper.writeValueAsString(ci));
					valueMap.put("dataset", openmlID);
					valueMap.put("evaluation_date",
							new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(Date.from(Instant.now())));
					valueMap.put("score", Double.toString(score));
					adapter.insert(config.getDBTableName(), valueMap);
					
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}