package dyadranking.performance;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;

import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;

public class PatternStochastics {

	public static void main(String... args) throws SQLException {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		String tableName = "dyad_dataset_support_5";

		// collect all patterns
		ResultSet rs = adapter.getResultsOfQuery("SELECT DISTINCT y FROM " + tableName);
		List<String> patterns = new ArrayList<>();
		while (rs.next()) {
			patterns.add(rs.getString(1));
		}

		// calculate statistics
		for (String pattern : patterns) {
			SummaryStatistics statistics = new SummaryStatistics();
			rs = adapter.getResultsOfQuery("SELECT score FROM " + tableName + " WHERE y='" + pattern + "'");
			while (rs.next()) {
				statistics.addValue(rs.getDouble(1));
			}
			System.out.println("Results for "+ pattern);
			System.out.println("Amount of values "+ statistics.getN());
			System.out.println("Avg score: "+ statistics.getMean());
			System.out.println("Std. Dev: "+ statistics.getStandardDeviation());
		}
	}

}
