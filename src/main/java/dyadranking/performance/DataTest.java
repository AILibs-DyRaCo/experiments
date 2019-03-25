//package dyadranking.performance;
//import java.io.File;
//import java.io.FileInputStream;
//import java.util.HashSet;
//import java.util.LinkedList;
//import java.util.Set;
//
//import org.junit.Test;
//
//import jaicore.ml.core.dataset.IInstance;
//import jaicore.ml.dyadranking.Dyad;
//import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
//import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
//import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
//import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
//import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;
//
//public class DataTest {
//
//	@Test
//	public void test() {
//
//		DyadRankingDataset trainData = new DyadRankingDataset();
//		DyadRankingDataset testData = new DyadRankingDataset();
//		PLNetDyadRanker ranker = new PLNetDyadRanker();
//
//		try {
//			trainData.deserialize(new FileInputStream(new File("train_data.txt")));
//			testData.deserialize(new FileInputStream(new File("test_data.txt")));
//
//			removeDuplicateEntries(testData);
//			removeDuplicateEntries(trainData);
//
//			System.out.println("train data");
//
//			for (IInstance instance : trainData)
//				System.out.println("double entries: " + countDuplicates((DyadRankingInstance) instance));
//
//			System.out.println("test data");			
//			
//			for (IInstance instance : testData)
//				System.out.println("double entries: " + countDuplicates((DyadRankingInstance) instance));
//
//			
//			
//			ranker.train(trainData);
//			System.out.println("avg kendalls tau: "
//					+ DyadRankingLossUtil.computeAverageLoss(new KendallsTauDyadRankingLoss(), testData, ranker));
//		} catch (Exception e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//	}
//
//	private static int countDuplicates(DyadRankingInstance instance) {
//		int result = instance.length();
//		HashSet<Dyad> dyads = new HashSet<Dyad>();
//		for (Dyad dyad : instance)
//			dyads.add(dyad);
//		result -= dyads.size();
//		return result;
//	}
//
//	private static void removeDuplicateEntries(DyadRankingDataset dataset) {
//		for (IInstance instance : dataset) {
//			DyadRankingInstance drInstance = (DyadRankingInstance) instance;
//			Set<Dyad> seenDyads = new HashSet<Dyad>();
//			LinkedList<Integer> removeIndices = new LinkedList<Integer>();
//			for (int i = drInstance.length() - 1; i >= 0; i--) {
//				Dyad curDyad = drInstance.getDyadAtPosition(i);
//				if (seenDyads.contains(curDyad)) {
//					removeIndices.add(i);
//				}
//				seenDyads.add(curDyad);
//			}
//			for(int index : removeIndices)
//				drInstance.removeDyadAtPosition(index);
//			
//			System.out.println(removeIndices);
//		}
//
//	}
//
//}
