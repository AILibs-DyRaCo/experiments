package dyadranking.experiments.util;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;

public class PLNetTrainer {
	
	private static final DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
	
	private static void printUsage() {
		System.out.println("Usage: trainingdata outputpath" );
	}
	
	public static void main(String[] args) {
		if (args.length != 2)
			printUsage();
		
		String dataPath = args[0];
		String outputPath = args[1];
		
		// Read training data
		File dataFile = new File(dataPath);
		if (!dataFile.exists())
			throw new IllegalArgumentException("Specified training data file path does not exist.");
		
		DyadRankingDataset data = new DyadRankingDataset();
		BufferedInputStream dataReader = null;		
		try {
			dataReader = new BufferedInputStream(new FileInputStream(dataFile));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		}
		data.deserialize(dataReader);
		
		// Train and save model
		PLNetDyadRanker plNet = new PLNetDyadRanker();
		plNet.printConfig();
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
