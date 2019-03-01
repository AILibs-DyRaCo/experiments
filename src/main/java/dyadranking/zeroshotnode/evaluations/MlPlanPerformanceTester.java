package dyadranking.zeroshotnode.evaluations;


import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Paths;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.PreferenceBasedNodeEvaluator;
import hasco.model.ComponentInstance;
import hasco.serialization.ComponentLoader;
import hasco.serialization.HASCOJacksonModule;

/**
 * Compares the performance results of ml-plan that uses the
 * {@link PreferenceBasedNodeEvaluator} against ml-plan that uses the
 * {@link DyadRankingBasedNodeEvaluator}.
 * 
 * 
 * Experiment setup:
 * 1) Pretrained plnet on our dyad-ranking dataset
 * 1.1) with landmarkers in the algorithm meta features
 * 1.2) w/o those
 * 
 * 
 * @author mirko
 *
 */
public class MlPlanPerformanceTester {

	public static void main(String... args) throws JsonParseException, JsonMappingException, IOException, URISyntaxException {
		String json = "{\"component\": {\"name\": \"weka.classifiers.meta.Stacking\", \"parameters\": [{\"name\": \"X\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"10\", \"defaultDomain\": {\"values\": [\"10\"]}}, {\"name\": \"S\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"1\", \"defaultDomain\": {\"values\": [\"1\"]}}], \"dependencies\": [], \"providedInterfaces\": [\"weka.classifiers.meta.Stacking\", \"AbstractClassifier\", \"MetaClassifier\", \"BaseClassifier\"], \"requiredInterfaces\": {}}, \"parameterValues\": {\"S\": \"1\", \"X\": \"10\"}, \"satisfactionOfRequiredInterfaces\": {}, \"parametersThatHaveBeenSetExplicitly\": [{\"name\": \"S\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"1\", \"defaultDomain\": {\"values\": [\"1\"]}}, {\"name\": \"X\", \"numeric\": false, \"categorical\": true, \"defaultValue\": \"10\", \"defaultDomain\": {\"values\": [\"10\"]}}], \"parametersThatHaveNotBeenSetExplicitly\": []}";
		ObjectMapper mapper = new ObjectMapper();
		File jsonFile = Paths
                .get(MlPlanPerformanceTester.class.getClassLoader().getResource(Paths.get("weka", "weka-all-autoweka.json").toString()).toURI())
                .toFile();
		mapper.registerModule(new HASCOJacksonModule(new ComponentLoader(jsonFile).getComponents()));
		ComponentInstance ci = mapper.readValue(json, ComponentInstance.class);
		System.out.println(ci);
	}

}
