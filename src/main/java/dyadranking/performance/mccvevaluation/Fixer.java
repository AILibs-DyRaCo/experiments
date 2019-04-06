package dyadranking.performance.mccvevaluation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.json.simple.JSONObject;

import com.fasterxml.jackson.databind.ObjectMapper;

import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.util.AbstractDyadScaler;
import jaicore.ml.dyadranking.util.DyadMinMaxScaler;

public class Fixer {

	private static String pathToConfigFolder = "/Users/elppa/git_pg/DRACO/approach_5_mccv/jsonConfigsMCCV";

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

		//	System.out.println("Reading " + potentiallyConfig.getName());
			JSONObject jsonObject = new ObjectMapper().readValue(potentiallyConfig, JSONObject.class);

			boolean hasNormalRanker = jsonObject.containsKey(JSONConfigKeys.RANKER_PATH);

			boolean hasScaledRanker = jsonObject.containsKey(JSONConfigKeys.RANKER_WITH_NORMALIZATION);

			if (hasNormalRanker && hasScaledRanker) {
				//System.out.println("Ranker was already trained on this dataset");
				// if (new File((String) jsonObject.get("rankerWithSer")).exists())
				//continue;
			}

			String datasetPath = (String) jsonObject.get(JSONConfigKeys.DATASET);

			String datasetWithNormalizationPath = (String) jsonObject.get(JSONConfigKeys.DATASET_NORM_KEY);

			int subsamplingSize = (int) jsonObject.get(JSONConfigKeys.SUBSAMPLING_SIZE);

			int splitIndex = (int) jsonObject.get(JSONConfigKeys.MCCV_INDEX);

		//	System.out.println("Found the following datasets: " + datasetPath + ", " + datasetWithNormalizationPath);

			String normalizerPath = (String) jsonObject.get(JSONConfigKeys.SCALER_PATH);
			
			DyadRankingDataset dataset = new DyadRankingDataset();
			dataset.deserialize(new FileInputStream(datasetPath));
			System.out.println(dataset.size());
			
			DyadRankingDataset dataset2 = new DyadRankingDataset();
			dataset2.deserialize(new FileInputStream(datasetWithNormalizationPath));
			
			AbstractDyadScaler minMaxScaler = new DyadMinMaxScaler();
			minMaxScaler.fit(dataset);
			minMaxScaler.transformAlternatives(dataset);
			//assert dataset.equals(dataset2);
			System.out.println(dataset.equals(dataset2));
			File normalizer = new File(normalizerPath);
			normalizer.getParentFile().mkdirs();
			System.out.println(normalizerPath);
			try(ObjectOutputStream oout = new ObjectOutputStream(new FileOutputStream(normalizerPath))){
				oout.writeObject(minMaxScaler);
				oout.flush();
			}catch(IOException e) {}
			
		}
	}
}
