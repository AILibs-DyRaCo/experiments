//package dyadranking.zeroshotml.datagen;
//
//import java.io.File;
//import java.io.FileInputStream;
//import java.io.FileNotFoundException;
//import java.io.FileOutputStream;
//import java.io.IOException;
//import java.io.ObjectInputStream;
//import java.io.ObjectOutputStream;
//
//import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
//import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
//import jaicore.ml.dyadranking.util.DyadNormalScaler;
//
//public class NormalScalerTrainer {
//	
//	public static String DATA_PATH = "datasets/zeroshot/";
//	
//	public static String TRAIN_FILE = "SMORBFtrain.dr";
//	
//	public static void main(String[] args) {
//		File dataTrainFile = new File(DATA_PATH + TRAIN_FILE);
//		DyadRankingDataset dataTrain = new DyadRankingDataset();
//		FileInputStream inpStream = null;
//		
//		try {
//			inpStream = new FileInputStream(dataTrainFile);
//			dataTrain.deserialize(inpStream);
//		} catch (FileNotFoundException e) {
//			e.printStackTrace();
//			System.exit(1);
//		} finally {
//			try {
//				inpStream.close();
//			} catch (IOException e) {
//				e.printStackTrace();
//				System.exit(1);
//			}
//		}
//		
//		DyadNormalScaler scaler = new DyadNormalScaler();
//		scaler.fit(dataTrain);
//		
//		File outputFile = new File(DATA_PATH + TRAIN_FILE + "_scaler");
//		FileOutputStream fileOut = null;
//		
//		try {
//			fileOut = new FileOutputStream(outputFile);
//			ObjectOutputStream objOut = new ObjectOutputStream(fileOut);
//			objOut.writeObject(scaler);
//			objOut.close();
//		} catch (IOException e) {
//			e.printStackTrace();
//			System.exit(1);
//		} finally {
//			try {
//				fileOut.close();
//			} catch (IOException e) {
//				e.printStackTrace();
//				System.exit(1);
//			}
//		}
//		
//		DyadNormalScaler scalerDeser = null;
//		
//		FileInputStream fileIn = null;
//		try {		
//			fileIn = new FileInputStream(outputFile);
//			ObjectInputStream objIn = new ObjectInputStream(fileIn);
//			scalerDeser = (DyadNormalScaler) objIn.readObject();
//			objIn.close();
//		} catch (ClassNotFoundException e) {
//			e.printStackTrace();
//		} catch (IOException e) {
//			e.printStackTrace();
//		} finally {
//			try {
//				fileIn.close();
//			} catch (IOException e) {
//				e.printStackTrace();
//				System.exit(1);
//			}
//		}
//		
//		for(int i = 0; i < 5; i++) {
//			IDyadRankingInstance testDr = dataTrain.get(i);
//			System.out.println(testDr);
//		}
//		System.out.println();
//		scaler.transform(dataTrain);
//		for(int i = 0; i < 5; i++) {
//			IDyadRankingInstance testDr = dataTrain.get(i);
//			System.out.println(testDr);
//		}
//		System.out.println();
//		scalerDeser.untransform(dataTrain);
//		for(int i = 0; i < 5; i++) {
//			IDyadRankingInstance testDr = dataTrain.get(i);
//			System.out.println(testDr);
//		}
//	}
//
//}
