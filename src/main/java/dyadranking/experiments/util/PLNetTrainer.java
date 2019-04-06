package dyadranking.experiments.util;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.core.predictivemodel.IPredictiveModelConfiguration;
import jaicore.ml.dyadranking.algorithm.IPLNetDyadRankerConfiguration;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.util.DyadStandardScaler;

public class PLNetTrainer {
	
	private static final DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
	
	private static void printUsage() {
		System.out.println("Usage: trainingdata outputpath preprocess[optional] \n"
				+ "preprocess can be one of { standardize, standardizeinstances, standardizealternatives, normalize, normalizeinstances, normalizealternatives }" );
	}
	
	/**
	 * Pretty prints a PLNet's configuration.
	 */
	public static void printConfig(IPredictiveModelConfiguration config) {
		IPLNetDyadRankerConfiguration configuration = (IPLNetDyadRankerConfiguration) config;
		StringBuilder output = new StringBuilder();
		output.append("PLNet config:")
			  .append("\n\tLearning rate:\t\t\t").append(configuration.plNetLearningRate())
			  .append("\n\tHidden nodes:\t\t\t").append(configuration.plNetHiddenNodes())
			  .append("\n\tSeed:\t\t\t\t").append(configuration.plNetSeed())
			  .append("\n\tActivation function:\t\t").append(configuration.plNetActivationFunction())
			  .append("\n\tMax epochs:\t\t\t").append(configuration.plNetMaxEpochs())
			  .append("\n\tMini batch size:\t\t").append(configuration.plNetMiniBatchSize())
			  .append("\n\tEarly stopping interval:\t").append(configuration.plNetEarlyStoppingInterval())
			  .append("\n\tEarly stopping patience:\t").append(configuration.plNetEarlyStoppingPatience())
			  .append("\n\tEarly stopping train ratio:\t").append(configuration.plNetEarlyStoppingTrainRatio())
			  .append("\n\tEarly stopping retrain:\t\t").append(configuration.plNetEarlyStoppingRetrain())
			  .append("\n");
		System.out.print(output.toString());
	}
	
	public static void main(String[] args) {
		if (args.length < 2)
			printUsage();
		
		String dataPath = args[0];
		String outputPath = args[1];
		
		// Read training data
		File dataFile = new File(dataPath);
		if (!dataFile.exists())
			throw new IllegalArgumentException("Specified training data file path does not exist.");
		
		DyadRankingDataset data = new DyadRankingDataset();
		FileInputStream dataReader = null;
		try {
			dataReader = new FileInputStream(dataFile);
			System.out.println("Loading data...");
			data.deserialize(dataReader);
			System.out.println("Data loaded.");
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		} finally {
			if (dataReader != null) {
				try {
					dataReader.close();
				} catch (IOException e) {
					e.printStackTrace();
					return;
				}
			}
		}

		if (args.length == 3) {
			System.out.println("Preprocessing...");
			String preprocess = args[2];
			DyadStandardScaler stdScaler = new DyadStandardScaler();
	//		DyadNormalScaler normScaler = new DyadNormalScaler();
			switch (preprocess.toLowerCase()) {
				case "standardize":
					stdScaler.fitTransform(data);
					break;
				case "standardizeinstances":
					stdScaler.fit(data);
					stdScaler.transformInstances(data);
					break;
				case "standardizealternatives":
					stdScaler.fit(data);
					stdScaler.transformAlternatives(data);
					break;
				case "normalize":
//					normScaler.fitTransform(data);
					break;
				case "normalizeinstances":
//					normScaler.fit(data);
//					normScaler.transformInstances(data);
					break;
				case "normalizealternatives":
//					normScaler.fit(data);
//					normScaler.transformAlternatives(data);
					break;			
			}			
			System.out.println("Finished preprocessing");
		}
		
		// Train and save model
		PLNetDyadRanker plNet = new PLNetDyadRanker();
		printConfig(plNet.getConfiguration());
		System.out.println("Beginning training at: " + dtf.format(LocalDateTime.now()));
		try {
			plNet.train(data);
		} catch (TrainingException e) {
			e.printStackTrace();
			return;
		}
		System.out.println("Training concluded at: " + dtf.format(LocalDateTime.now()));
		try {
			plNet.saveModelToFile(outputPath);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
