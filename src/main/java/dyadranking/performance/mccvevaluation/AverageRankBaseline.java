package dyadranking.performance.mccvevaluation;

import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.stream.Collectors;

import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.dyadranking.Dyad;

public class AverageRankBaseline {

	private Map<String, List<DatabaseDyad>> cachedDyads;

	public AverageRankBaseline(Map<String, List<DatabaseDyad>> cachedDyads) {
		this.cachedDyads = cachedDyads;
	}

	/**
	 * Average rank approach: For each dyad it takes the average performance score
	 * form the train datasets and ranks the dyad according to this score.
	 * 
	 * @param datasets
	 *            the train datasets
	 * @param adapter
	 * @param toRank
	 *            the dyads to rank
	 * @return
	 * @throws SQLException
	 */
	public List<Pair<Double, Dyad>> getAverageRankForDatasets(List<Dyad> toRank, String key) {
		List<Pair<Double, Dyad>> toReturn = new ArrayList<>();
		List<DatabaseDyad> cache = cachedDyads.get(key);
		for (Dyad dyad : toRank) {
			String serializedY = dyad.getAlternative().stream().boxed().map(d -> d.toString())
					.collect(Collectors.joining(" "));
			OptionalDouble optScore = cache.stream().filter(d -> d.getSerializedY().equals(serializedY))
					.mapToDouble(DatabaseDyad::getScore).average();
			if (optScore.isPresent()) {
				// there are some pipelines missing
				toReturn.add(new Pair<>(optScore.getAsDouble(), dyad));
			} else {
				toReturn.add(new Pair<>(1.0, dyad));
			}
		}
		Collections.sort(toReturn, DyadComparator::compare);
		return toReturn;
	}

}
