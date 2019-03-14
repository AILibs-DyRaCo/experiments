package dyadranking.performance.mccvevaluation;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.json.simple.JSONObject;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

public class JSONConfigCleaner {
	
	private static String pathToConfigFolder;
	
	public static void main(String[] args) throws JsonParseException, JsonMappingException, IOException {
		if (args.length > 0 && pathToConfigFolder == null) {
			pathToConfigFolder = args[0];
		}

		File jsonFolder = new File(pathToConfigFolder);
		File[] configs = jsonFolder.listFiles();

		List<File> shuffeled = Arrays.asList(configs);
		Collections.shuffle(shuffeled);
		for (File potentiallyConfig : shuffeled) {
			if (potentiallyConfig.isDirectory() || !potentiallyConfig.getName().contains(".json")) {
				continue;
			}

			System.out.println("Reading " + potentiallyConfig.getName());
			JSONObject jsonObject = new ObjectMapper().readValue(potentiallyConfig, JSONObject.class);

			if (jsonObject.containsKey(JSONConfigKeys.RANKER_WITH_NORMALIZATION)) {
				System.out.println("Ranker was already trained on this dataset");
				if (!new File((String) jsonObject.get(JSONConfigKeys.RANKER_WITH_NORMALIZATION)).exists()) {
					jsonObject.remove(JSONConfigKeys.RANKER_WITH_NORMALIZATION);
				}
			}
			
			if (jsonObject.containsKey(JSONConfigKeys.RANKER_PATH)) {
				if (!new File((String) jsonObject.get(JSONConfigKeys.RANKER_PATH)).exists()) {
					jsonObject.remove(JSONConfigKeys.RANKER_PATH);
				}
			}
			
			potentiallyConfig.delete();
			try (FileWriter writer = new FileWriter(potentiallyConfig)) {
				writer.write(jsonObject.toJSONString());
				writer.flush();
			} catch (IOException e) {
				System.err.println("Failed to write json");
			}
		}
	}
}
