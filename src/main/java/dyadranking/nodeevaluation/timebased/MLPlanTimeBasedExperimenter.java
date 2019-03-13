package dyadranking.nodeevaluation.timebased;

import java.io.File;
import java.io.IOException;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.aeonbits.owner.ConfigFactory;
import org.apache.tools.ant.BuildListener;

import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;

import de.upb.crc901.mlplan.core.MLPlan;
import de.upb.crc901.mlplan.core.MLPlanBuilder;
import de.upb.crc901.mlplan.dyadranking.DyadRankingBasedNodeEvaluator;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import dyadranking.sql.SQLUtils;
import hasco.events.HASCOSolutionEvent;
import hasco.serialization.ComponentLoader;
import jaicore.basic.SQLAdapter;
import jaicore.basic.TimeOut;
import jaicore.ml.WekaUtil;
import jaicore.ml.cache.ReproducibleInstances;
import jaicore.planning.hierarchical.algorithms.forwarddecomposition.graphgenerators.tfd.TFDNode;
import jaicore.search.algorithms.standard.bestfirst.events.EvaluatedSearchSolutionCandidateFoundEvent;
import jaicore.search.algorithms.standard.bestfirst.events.FValueEvent;
import jaicore.search.model.travesaltree.Node;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

/**
 * Used for the Evaluations from Mirkos Paper: Requires AILibs branch:
 * mirkos-experiments (which somehow destroys MLPlans architecture to get a
 * better view)
 * 
 * 
 * Simply conducts MLPlan and listens to the Evaluation Events of the underlying
 * HASCO-search.
 * 
 * @author mirkoj
 *
 */
public class MLPlanTimeBasedExperimenter {

	public static void main(String[] args) throws Exception {

		MLPlanTimeBasedExperimenterConfig config = ConfigFactory.create(MLPlanTimeBasedExperimenterConfig.class);

		SQLAdapter sqlAdapter = SQLUtils.sqlAdapterFromArgs(args);

		// createTable(sqlAdapter, "dyadranking_versus_mlplan");
		// shameless copy of the MLPlanOpenML example...
		int experimentCounter = 0;
		for (int dataset : config.getOpenMLIds()) {
			for (long timeout : config.getTimeouts()) {
				for (int seed : config.getSeeds()) {
					String table ="";
					if (config.useDyadRanking())
						table = "dyadranking_versus_mlplan_approach_5";
					else
						table = "dyadranking_versus_mlplan_approach_5_2";
					
					MLPLanSolutionLogger logger = new MLPLanSolutionLogger(sqlAdapter, seed, timeout, dataset,
							table);

					ReproducibleInstances data = ReproducibleInstances.fromOpenML(Integer.toString(dataset),
							"4350e421cdc16404033ef1812ea38c01");
					data.setClassIndex(data.numAttributes() - 1);
					List<Instances> split = WekaUtil.getStratifiedSplit((Instances) data, (new Random(0)).nextLong(),
							0.7d);

					MLPlanBuilder builder = new MLPlanBuilder();

					builder.withSearchSpaceConfigFile(
							new File("conf/automl/searchmodels/weka/weka-approach-5-autoweka.json"));
					if (config.useDyadRanking()) {
						builder = builder.withAutoWEKAConfiguration(false);
						builder.withPreferredNodeEvaluator(new DyadRankingBasedNodeEvaluator<>(new ComponentLoader(
								new File("conf/automl/searchmodels/weka/weka-approach-5-autoweka.json"))));
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
					logger.startTime();
					mlplan.registerListener(logger);
					System.out.println("starting");
					Classifier optimizedClassifier = mlplan.call();
					long trainTime = Duration.between(logger.startTime, Instant.now()).toMillis();
					System.out.println("Finished build of the classifier.");
					System.out.println("Chosen model is: " + ((MLPipeline) mlplan.getSelectedClassifier()).toString());
					System.out.println("Training time was " + trainTime + "s.");

					/* evaluate solution produced by mlplan */
					Evaluation eval = new Evaluation(split.get(0));
					eval.evaluateModel(optimizedClassifier, split.get(1));
					System.out.println("Error is " + eval.errorRate());
				}
			}
		}
	}

	private static void createTable(SQLAdapter adapter, String tableName) throws SQLException {
		ResultSet rs = adapter.getResultsOfQuery("SHOW TABLES");
		boolean hasPerformanceTable = false;
		while (rs.next()) {
			String ptableName = rs.getString(1);
			if (ptableName.equals(tableName))
				hasPerformanceTable = true;
		}

		if (!hasPerformanceTable) {
			adapter.update("CREATE TABLE " + tableName + " (`id` int(10) NOT NULL AUTO_INCREMENT,\r\n"
					+ " `dataset` int(10) NOT NULL, \r \n" + " `seed` int(10) NOT NULL,\r\n"
					+ " `timeout` int(10) NOT NULL,\r\n" + " `scoreOfSolution` double NOT NULL, \r \n"
					+ " `isFValue` BIT NOT NULL,\r\n"
					+ "`timeUntilFound` double NOT NULL, PRIMARY KEY(id)) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin",
					new ArrayList<>());
		}
	}

	private static class MLPLanSolutionLogger {

		private SQLAdapter adapter;
		private int dataset;
		private long timeout;
		private Instant startTime;
		private String tableName;
		private int seed;

		public MLPLanSolutionLogger(SQLAdapter adapter, int seed, long timeout, int dataset, String tableName) {
			this.adapter = adapter;
			this.tableName = tableName;
			this.timeout = timeout;
			this.dataset = dataset;
			this.seed = seed;
			try {
				createTable(adapter, tableName);
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		public void startTime() {
			startTime = Instant.now();
		}

		/**
		 * This listener listens to the search of HASCO! Thus, it will see any
		 * random-completion that was evaluated (this is not the F-Value!)
		 * 
		 * @param randomCompletion
		 */
		@AllowConcurrentEvents
		@Subscribe
		public void listenToRandomCompleterSolution(
				EvaluatedSearchSolutionCandidateFoundEvent<Node<TFDNode, String>, String, Double> randomCompletion) {
			Duration timeUntilFound = Duration.between(startTime, Instant.now());
			Map<String, Object> values = new HashMap<>();
			values.put("timeUntilFound", new Double(timeUntilFound.toMillis()));
			values.put("scoreOfSolution", (Double) randomCompletion.getSolutionCandidate().getScore());
			values.put("dataset", this.dataset);
			values.put("timeout", (double) this.timeout);
			values.put("seed", this.seed);
			// these are just intermediate solutions
			values.put("isFValue", 0);
			try {
				adapter.insert(tableName, values);
			} catch (SQLException e) {
				e.printStackTrace();
			}
		}

		/**
		 * This listener listens to the search of HASCO! These, events will be thrown by
		 * HASCO after the node was evaluated.
		 * 
		 * @param fValueSolution
		 */
		@AllowConcurrentEvents
		@Subscribe
		public void listenToHASCOSolution(FValueEvent<Double> fValueSolution) {
			Duration timeUntilFound = Duration.between(startTime, Instant.now());
			Map<String, Object> values = new HashMap<>();
			values.put("timeUntilFound", new Double(timeUntilFound.toMillis()));
			values.put("scoreOfSolution", fValueSolution.getfValue());
			values.put("dataset", this.dataset);
			values.put("timeout", (double) this.timeout);
			values.put("isFValue", 1.0d);
			values.put("seed", this.seed);
			try {
				adapter.insert(tableName, values);
			} catch (SQLException e) {
				e.printStackTrace();
			}

		}
	}

}
