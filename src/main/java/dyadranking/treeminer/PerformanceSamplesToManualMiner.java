package dyadranking.treeminer;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.semanticweb.owlapi.model.OWLOntologyCreationException;

import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.crc901.mlplan.metamining.pipelinecharacterizing.ComponentInstanceVectorFeatureGenerator;
import de.upb.isys.linearalgebra.DenseDoubleVector;
import dyadranking.sql.SQLUtils;
import hasco.model.ComponentInstance;
import hasco.serialization.ComponentLoader;
import hasco.serialization.HASCOJacksonModule;
import jaicore.basic.SQLAdapter;

/**
 * Extracts all ComponentInstances from the performance database and mines the
 * patterns.
 * 
 * @author mirkoj
 *
 */
public class PerformanceSamplesToManualMiner {

	public static void main(String... args)
			throws IOException, URISyntaxException, InterruptedException, SQLException, OWLOntologyCreationException {

		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);

		String resultTableName = "dyad_dataset_approach_5_performance_samples_with_SMO";

		ObjectMapper mapper = new ObjectMapper();
		File jsonFile = Paths.get(ComponentInstanceVectorFeatureGenerator.class.getClassLoader()
				.getResource(Paths.get("weka", "weka-approach-5-autoweka.json").toString()).toURI())
				.toFile();
		ComponentLoader loader = new ComponentLoader(jsonFile);
		mapper.registerModule(new HASCOJacksonModule(loader.getComponents()));

		ComponentInstanceVectorFeatureGenerator characterizer = new ComponentInstanceVectorFeatureGenerator(loader.getComponents());

		// create the result table
		createDyadTable(adapter, resultTableName);

		// select the average score for a distinct pipeline on a dataset
		ResultSet pipelinesGroupedByDataset = adapter.getResultsOfQuery(
				"SELECT composition, dataset_id, loss, pipeline_id FROM `pipeline_performance_5_classifiers_with_SMO` NATURAL JOIN draco_pipelines_5_classifiers_with_SMO WHERE loss IS NOT NULL AND pipeline_id >= 3520");

		System.out.println("Query done");
		while (pipelinesGroupedByDataset.next()) {
			String composition = pipelinesGroupedByDataset.getString(1);
			int dataset = Integer.parseInt(pipelinesGroupedByDataset.getString(2).replaceAll("\"", ""));
			double avgScore = pipelinesGroupedByDataset.getDouble(3);
			int pipelineId = pipelinesGroupedByDataset.getInt(4);
			// deserialize component-instance
			ComponentInstance cI = mapper.readValue(composition, ComponentInstance.class);
			double[] y = characterizer.characterize(cI, new DenseDoubleVector(characterizer.getLengthOfCharacterization(), 0));
			String serializedY = Arrays.stream(y).mapToObj(d -> d + "").collect(Collectors.joining(" "));

			Map<String, Object> values = new HashMap<>();
			values.put("dataset", dataset);
			values.put("y", serializedY);
			values.put("score", avgScore);
			values.put("pipeline_id", pipelineId);
			adapter.insert(resultTableName, values);

		}

	}

	private static void createDyadTable(SQLAdapter adapter, String newTableName) throws SQLException {
		ResultSet rs = adapter.getResultsOfQuery("SHOW TABLES");
		boolean hasPerformanceTable = false;
		while (rs.next()) {
			String tableName = rs.getString(1);
			if (tableName.equals(newTableName))
				hasPerformanceTable = true;
		}

		if (!hasPerformanceTable) {
			adapter.update(
					"CREATE TABLE " + newTableName + " (\r\n" + " `id` int(10) NOT NULL AUTO_INCREMENT,\r\n"
							+ " `dataset` TEXT NOT NULL, \r \n" + " `score` double NOT NULL,\r\n" + " `y` TEXT, \r \n"
							+ "`pipeline_id` int(10) NOT NULL, \r \n" + " PRIMARY KEY (`id`)\r\n"
							+ ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COLLATE=utf8_bin",
					new ArrayList<>());
		}
	}
}
