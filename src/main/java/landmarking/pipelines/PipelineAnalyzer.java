package landmarking.pipelines;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;

import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;

public class PipelineAnalyzer {

	public static void main(String args[]) {

		HashMap<String, Integer> uniquePipelines = new HashMap<String, Integer>();
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);

		String query = "SELECT pipeline_id, COUNT(dataset_id), GROUP_CONCAT(loss ORDER BY dataset_id) FROM `pipeline_landmarking_v2`WHERE loss is not null GROUP BY pipeline_id  \n"
				+ "ORDER BY COUNT(dataset_id)  DESC";

		try {
			ResultSet rs = adapter.getResultsOfQuery(query);
			int counter = 0;
			while (rs.next() && counter < 250) {
				System.out.println(rs.getInt(1) + "\t" + rs.getInt(2) + "\t" + rs.getString(3));
				if (rs.getInt(2) == 8)
					uniquePipelines.put(rs.getString(3), rs.getInt(1));
				counter++;
			}
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		for (int pipelineID : uniquePipelines.values()) {
			try {
				ResultSet rs = adapter.getResultsOfQuery(
						"SELECT mlpipeline FROM draco_pipelines WHERE pipeline_id = " + Integer.toString(pipelineID));
				rs.first();
				System.out.println(rs.getString(1));
			} catch (SQLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
}
