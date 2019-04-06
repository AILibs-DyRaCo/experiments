package dyadranking.performance.mccvevaluation;

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

			boolean hasNormalRanker = jsonObject.containsKey(JSONConfigKeys.RANKER_PATH);

		//	boolean hasScaledRanker = jsonObject.containsKey(JSONConfigKeys.RANKER_WITH_NORMALIZATION);

			boolean hasScaledRanker = true;
			
			if (hasNormalRanker && hasScaledRanker) {
				System.out.println("Ranker was already trained on this dataset");
				// if (new File((String) jsonObject.get("rankerWithSer")).exists())
				continue;
			}

			String datasetPath = (String) jsonObject.get(JSONConfigKeys.DATASET);

			String datasetWithNormalizationPath = (String) jsonObject.get(JSONConfigKeys.DATASET_NORM_KEY);

			int subsamplingSize = (int) jsonObject.get(JSONConfigKeys.SUBSAMPLING_SIZE);

			int splitIndex = (int) jsonObject.get(JSONConfigKeys.MCCV_INDEX);

			System.out.println("Found the following datasets: " + datasetPath + ", " + datasetWithNormalizationPath);

			File rankerOutFolder = new File("rankersMCCV");
			if (!rankerOutFolder.exists()) {
				rankerOutFolder.mkdir();
			}

			System.out.println("Updating json");

			if (!hasNormalRanker) {
				jsonObject.put("ranker", new File("rankersMCCV/ranker_" + subsamplingSize + "_" + splitIndex + ".zip")
						.getAbsolutePath());
			}

			if (!hasScaledRanker) {
				jsonObject.put("rankerWithSer",
						new File("rankersMCCV/ranker_with_norm_" + subsamplingSize + "_" + splitIndex + ".zip")
								.getAbsolutePath());
			}

			potentiallyConfig.delete();
			try (FileWriter writer = new FileWriter(potentiallyConfig)) {
				writer.write(jsonObject.toJSONString());
				writer.flush();
			} catch (IOException e) {
				System.err.println("Failed to write json");
			}
			PLNetDyadRanker ranker;
			DyadRankingDataset dataset;

			if (!hasNormalRanker) {
				ranker = new PLNetDyadRanker();

				dataset = new DyadRankingDataset();
				dataset.deserialize(new FileInputStream(new File(datasetPath)));
				ranker.train(dataset);
				System.out.println("Finished Training non-normalized ranker...");
				ranker.saveModelToFile("rankersMCCV/ranker_" + subsamplingSize + "_" + splitIndex);
			}
			
			if (!hasScaledRanker) {
				dataset = new DyadRankingDataset();
				ranker = new PLNetDyadRanker();
				dataset.deserialize(new FileInputStream(new File(datasetWithNormalizationPath)));
				ranker.train(dataset);
				System.out.println("Finished training normalized ranker...");
				ranker.saveModelToFile("rankersMCCV/ranker_with_norm_" + subsamplingSize + "_" + splitIndex);
			}
		}
	}
}