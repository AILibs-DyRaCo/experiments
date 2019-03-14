package dyadranking.performance.mccvevaluation;

import java.util.Comparator;

import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.dyadranking.Dyad;

public class DyadComparator {

	public static int compare(Pair<Double, Dyad> o1, Pair<Double, Dyad> o2) {
		return Double.compare(o1.getX(), o2.getX());
	}

}
