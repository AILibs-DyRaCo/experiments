package dyadranking.performance;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.json.simple.JSONObject;

import com.fasterxml.jackson.databind.ObjectMapper;

import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;

/**
 * Reads the jsonConfigs and trains a PLNet for every config that can be found.
 * Finally, it extends the config with a path to the serialized ranker.
 *
 */
public class JsonConfigBasedPLNetTrainer {

	private static String pathToConfigFolder = null;

	public static void main(String[] args) throws Exception {
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

			if (jsonObject.containsKey("rankerWithSer")) {
				System.out.println("Ranker was already trained on this dataset");
				//if (new File((String) jsonObject.get("rankerWithSer")).exists())
				continue;
			}

			String datasetPath = (String) jsonObject.get("datasetPath");

			String datasetWithNormalizationPath = (String) jsonObject.get("datasetWithNorm");

			int subsamplingSize = (int) jsonObject.get("subsamplingSize");

			System.out.println("Found the following datasets: " + datasetPath + ", " + datasetWithNormalizationPath);

			File rankerOutFolder = new File("rankers");
			if (!rankerOutFolder.exists()) {
				rankerOutFolder.mkdir();
			}

			System.out.println("Updating json");

	//		jsonObject.put("ranker", new File("rankers/ranker_" + subsamplingSize + ".zip").getAbsolutePath());
			jsonObject.put("rankerWithSer",
					new File("rankers/ranker_with_norm_" + subsamplingSize + ".zip").getAbsolutePath());

			potentiallyConfig.delete();
			try (FileWriter writer = new FileWriter(potentiallyConfig)) {
				writer.write(jsonObject.toJSONString());
				writer.flush();
			} catch (IOException e) {
				System.err.println("Failed to write json");
			}
			PLNetDyadRanker ranker;
			DyadRankingDataset dataset;
			
//			if (!rankerExists) {
//				ranker = new PLNetDyadRanker();
//
//				dataset = new DyadRankingDataset();
//				dataset.deserialize(new FileInputStream(new File(datasetPath)));
//				ranker.train(dataset);
//				System.out.println("Finished Training non-normalized ranker...");
//				ranker.saveModelToFile("rankers/ranker_" + subsamplingSize);
//			}
			dataset = new DyadRankingDataset();
			ranker = new PLNetDyadRanker();

//			System.gc();

			dataset.deserialize(new FileInputStream(new File(datasetWithNormalizationPath)));
			ranker.train(dataset);
			System.out.println("Finished training normalized ranker...");
			ranker.saveModelToFile("rankers/ranker_with_norm_" + subsamplingSize);
		}
	}
}