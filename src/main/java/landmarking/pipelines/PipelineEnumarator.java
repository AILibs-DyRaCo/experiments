package landmarking.pipelines;

import java.io.File;
import java.io.FileReader;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.util.Pair;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.crc901.mlplan.core.MLPlan;
import de.upb.crc901.mlplan.core.MLPlanBuilder;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import dyadranking.sql.SQLUtils;
import hasco.core.Util;
import hasco.model.Component;
import hasco.model.ComponentInstance;
import jaicore.basic.SQLAdapter;
import jaicore.ml.evaluation.evaluators.weka.SingleRandomSplitClassifierEvaluator;
import jaicore.planning.hierarchical.algorithms.forwarddecomposition.graphgenerators.tfd.TFDNode;
import jaicore.search.algorithms.standard.bestfirst.events.GraphSearchSolutionCandidateFoundEvent;
import jaicore.search.algorithms.standard.random.RandomSearch;
import jaicore.search.core.interfaces.GraphGenerator;
import jaicore.search.model.other.SearchGraphPath;
import jaicore.search.probleminputs.GraphSearchInput;
import jaicore.search.structure.graphgenerator.SingleRootGenerator;
import weka.core.Instances;

public class PipelineEnumarator {

	private static int NUMBER_COMPLETIONS = 100;

	private static String DB_TABLE_NAME = "draco_pipelines_5_classifiers_with_SMO";
	
	private static String DATASET_PATH = "../experiments/datasets/toydataset.arff";

	private static String openMLKey = "4350e421cdc16404033ef1812ea38c01";
	private static int openMLDatasetID = 40983;

	private static String search1 = "weka.attributeSelection.Ranker";
	private static String evaluation1 = "weka.attributeSelection.CorrelationAttributeEval";

	private static String search2 = "weka.attributeSelection.BestFirst";
	private static String evaluation2 = "weka.attributeSelection.CfsSubsetEval";

	private static String search3 = "weka.attributeSelection.Ranker";
	private static String evaluation3 = "weka.attributeSelection.GainRatioAttributeEval";

	private static String search4 = "weka.attributeSelection.GreedyStepwise";
	private static String evaluation4 = "weka.attributeSelection.CfsSubsetEval";

	private static String search5 = "weka.attributeSelection.Ranker";
	private static String evaluation5 = "weka.attributeSelection.InfoGainAttributeEval";

	private static String search6 = "weka.attributeSelection.Ranker";
	private static String evaluation6 = "weka.attributeSelection.OneRAttributeEval";

	private static String search7 = "weka.attributeSelection.Ranker";
	private static String evaluation7 = "weka.attributeSelection.PrincipalComponents";

	private static String search8 = "weka.attributeSelection.Ranker";
	private static String evaluation8 = "weka.attributeSelection.ReliefFAttributeEval";

	private static String search9 = "weka.attributeSelection.Ranker";
	private static String evaluation9 = "weka.attributeSelection.SymmetricalUncertAttributeEval";

	public static void main(String args[]) {
		
		SQLAdapter sqlAdapter = SQLUtils.sqlAdapterFromArgs(args);

		/* initialize tables if not existent */
		try {
			ResultSet rs = sqlAdapter.getResultsOfQuery("SHOW TABLES");
			boolean hasPerformanceTable = false;
			while (rs.next()) {
				String tableName = rs.getString(1);
				if (tableName.equals(DB_TABLE_NAME))
					hasPerformanceTable = true;
			}

			// if there is no performance table, create it. we hash the composition and
			// trajectory and use the hash value as primary key for performance reasons.
			if (!hasPerformanceTable) {
				sqlAdapter.update("CREATE TABLE `" + DB_TABLE_NAME + "` (\r\n"
						+ " `pipeline_id` int(10) NOT NULL AUTO_INCREMENT,\r\n" + " `composition` json NOT NULL,\r\n"
						+ " `mlpipeline` VARCHAR(1000) NOT NULL,\r\n" + " PRIMARY KEY (`pipeline_id`)\r\n"
						+ ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COLLATE=utf8_bin", new ArrayList<>());
			}

		} catch (SQLException e) {
			e.printStackTrace();
		}

		try {
			
			Instances data = new Instances(new FileReader(new File(DATASET_PATH)));
			data.setClassIndex(data.numAttributes()-1);
			
			System.out.println(data);
			
			MLPlanBuilder builder = new MLPlanBuilder();
			builder.withAutoWEKAConfiguration();
			Collection<Component> components = new ArrayList<Component>(builder.getComponents());

			// Create all possible pipeline combinations
			List<Component> classifiers = new ArrayList<Component>();
			HashMap<String, Component> evaluators = new HashMap<String, Component>();
			HashMap<String, Component> searchers = new HashMap<String, Component>();
			Component pipelineComponent = null;
			Component attributeSelectionComponent = null;
			Component smoComponent = null;
			List<Component> kernelFunctions = new ArrayList<Component>();

			for (Component component : components) {
				if (component.getName().contains("classifier")) {
					if(component.getName().contains("functions.supportVector"))
						kernelFunctions.add(component);
					if(component.getName().contains("SMO"))
						smoComponent = component;
					classifiers.add(component);
					
				}
				if (component.getProvidedInterfaces().contains("evaluator"))
					evaluators.put(component.getName(), component);
				if (component.getProvidedInterfaces().contains("searcher"))
					searchers.put(component.getName(), component);
				if (component.getName().equals("pipeline"))
					pipelineComponent = component;
				if (component.getName().equals("weka.attributeSelection.AttributeSelection"))
					attributeSelectionComponent = component;
			}

			components.removeAll(classifiers);
			components.removeAll(evaluators.values());
			components.removeAll(searchers.values());

			System.out.println("Remaining components:");
			for (Component comp : components)
				System.out.println(comp.toString());

			List<Pair<Component, Component>> evaluatorSearcherPairs = new ArrayList<Pair<Component, Component>>();
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation1), searchers.get(search1)));
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation2), searchers.get(search2)));
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation3), searchers.get(search3)));
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation4), searchers.get(search4)));
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation5), searchers.get(search5)));
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation6), searchers.get(search6)));
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation7), searchers.get(search7)));
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation8), searchers.get(search8)));
			evaluatorSearcherPairs
					.add(new Pair<Component, Component>(evaluators.get(evaluation9), searchers.get(search9)));

			System.out.println("\nClassifiers:");
			for (Component comp : classifiers)
				System.out.println(comp.getName());
			System.out.println("\nEvaluators:");
			for (Component comp : evaluators.values())
				System.out.println(comp.getName());
			System.out.println("\nSearchers:");
			for (Component comp : searchers.values())
				System.out.println(comp.getName());
			System.out.println("\nLegal Pairs:");
			for (Pair<Component, Component> pair : evaluatorSearcherPairs)
				System.out.println(pair.toString());
			System.out.println("\nKernel Functions:");
			for (Component comp : kernelFunctions)
				System.out.println(comp.toString());
			System.out.println("\nSMO:");
			System.out.println(smoComponent);
			
			List<Collection<Component>> smoCombinations = new ArrayList<Collection<Component>>();		
			for(Component kernel : kernelFunctions) {
				ArrayList<Component> onlySMO = new ArrayList<Component>();
				onlySMO.add(kernel);
				onlySMO.add(smoComponent);
				smoCombinations.add(onlySMO);
//				onlySMO.add(pipelineComponent);
//				onlySMO.add(attributeSelectionComponent);
				for (Pair<Component, Component> pair : evaluatorSearcherPairs) {
					ArrayList<Component> preprocessorClassifierCombination = new ArrayList<Component>();
					preprocessorClassifierCombination.add(smoComponent);
					preprocessorClassifierCombination.add(kernel);
					preprocessorClassifierCombination.add(pair.getFirst());
					preprocessorClassifierCombination.add(pair.getSecond());
					preprocessorClassifierCombination.add(pipelineComponent);
					preprocessorClassifierCombination.add(attributeSelectionComponent);
					smoCombinations.add(preprocessorClassifierCombination);
				}
			}

//			List<Collection<Component>> allPipelineCombinations = new ArrayList<Collection<Component>>();
//			for (Component classifier : classifiers) {
//				ArrayList<Component> onlyClassifier = new ArrayList<Component>();
//				onlyClassifier.add(classifier);
//				onlyClassifier.add(pipelineComponent);
//				allPipelineCombinations.add(onlyClassifier);
//				for (Pair<Component, Component> pair : evaluatorSearcherPairs) {
//					ArrayList<Component> preprocessorClassifierCombination = new ArrayList<Component>();
//					preprocessorClassifierCombination.add(classifier);
//					preprocessorClassifierCombination.add(pair.getFirst());
//					preprocessorClassifierCombination.add(pair.getSecond());
//					preprocessorClassifierCombination.add(pipelineComponent);
//					preprocessorClassifierCombination.add(attributeSelectionComponent);
//					allPipelineCombinations.add(preprocessorClassifierCombination);
//				}
//			}

			for (Collection<Component> currentComponents : smoCombinations) {
				int numExceptions = 0;
				MLPlanBuilder builder1 = new MLPlanBuilder();
				try {
					builder1.withAutoWEKAConfiguration(currentComponents);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				SingleRandomSplitClassifierEvaluator eval = new SingleRandomSplitClassifierEvaluator(data);
				MLPlan currentMLPlan = new MLPlan(builder1, data);
				GraphGenerator gg = currentMLPlan.getGraphGenerator();
				GraphSearchInput gsi = new GraphSearchInput(gg);
				RandomSearch rs = new RandomSearch(gsi, 187);
				SingleRootGenerator<TFDNode> rg = (SingleRootGenerator<TFDNode>) gg.getRootGenerator();
				// if there are no parameters, we don't need several completions
				int repetitions = getNumberParamsOfPipeline(components) < 1 ? 1 : NUMBER_COMPLETIONS;
				int completions = 0;
				System.out.println("\n\nRandom completions for " + builder1.getComponents() + " with "
						+ repetitions + "repetitions:");
				while (completions < repetitions && rs.hasNext() && numExceptions < 150) {
					try {
						GraphSearchSolutionCandidateFoundEvent<?, ?, ?> ev = (GraphSearchSolutionCandidateFoundEvent<?, ?, ?>) rs.nextWithException();
						SearchGraphPath sgp = ev.getSolutionCandidate();
						TFDNode goalNode = (TFDNode) sgp.getNodes().get(sgp.getNodes().size() - 1);
						WEKAPipelineFactory factory = new WEKAPipelineFactory();
						ComponentInstance ci = Util.getSolutionCompositionFromState(builder1.getComponents(),
								goalNode.getState(), true);
						MLPipeline mlp = factory.getComponentInstantiation(ci);
						// if we are using preprocessing, make sure to have pipelines which actually use
						// preprocessors and don't leave them out
						if (currentComponents.size() > 2 && mlp.getPreprocessors().isEmpty()) {
							continue;
						}
						double loss = eval.evaluate(mlp);
						System.out.println(mlp.toString() + loss);
						ObjectMapper mapper = new ObjectMapper();
						String compositionString = mapper.writeValueAsString(ci);
						Map<String, String> valueMap = new HashMap<>();
						valueMap.put("composition", compositionString);
						valueMap.put("mlpipeline", mlp.toString());
						System.out.println("size:" + currentComponents.size());
//						System.out.println(mlp);
						completions++;
						sqlAdapter.insert(DB_TABLE_NAME, valueMap);
					} catch (Exception e) {
						e.printStackTrace();
						numExceptions++;
						continue;
					}
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		sqlAdapter.close();
	}

	private static int getNumberParamsOfPipeline(Collection<Component> components) {
		int sum = 0;
		for (Component comp : components)
			sum += comp.getParameters().size();
		return sum;
	}
}
