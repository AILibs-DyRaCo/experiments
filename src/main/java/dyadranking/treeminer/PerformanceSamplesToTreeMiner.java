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
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.semanticweb.owlapi.model.OWLOntologyCreationException;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import de.upb.crc901.mlplan.metamining.pipelinecharacterizing.WEKAPipelineCharacterizer;
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
public class PerformanceSamplesToTreeMiner {

	public static void main(String... args)
			throws IOException, URISyntaxException, InterruptedException, SQLException, OWLOntologyCreationException {
		String resultTableName = "dyad_dataset_new";
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);

		ObjectMapper mapper = new ObjectMapper();
		mapper.registerModule(new HASCOJacksonModule());

		Class<? extends PerformanceSamplesToTreeMiner> clazz = new PerformanceSamplesToTreeMiner().getClass();
		File jsonFile = Paths
				.get(clazz.getClassLoader().getResource(Paths.get("weka", "weka-all-autoweka.json").toString()).toURI())
				.toFile();

		ComponentLoader loader = new ComponentLoader(jsonFile);

		WEKAPipelineCharacterizer characterizer = minePatterns(adapter, loader, mapper);

		
		//create the result table
		createDyadTable(adapter, resultTableName);
		
		// select the average score for a distinct pipeline on a dataset
		ResultSet pipelinesGroupedByDataset = adapter.getResultsOfQuery(
				"SELECT COUNT(*) AS 'count', composition, AVG(score) AS 'AVG_SCORE', JSON_EXTRACT(train_trajectory, '$[0].inputs.id') AS 'dataset' from performance_samples GROUP BY composition, dataset HAVING count=10 ");

		while (pipelinesGroupedByDataset.next()) {
			String composition = pipelinesGroupedByDataset.getString(2);
			double avgScore = pipelinesGroupedByDataset.getDouble(3);
			int dataset = Integer.parseInt( pipelinesGroupedByDataset.getString(4).replaceAll("\"", ""));
			
			//deserialize component-instance
			ComponentInstance cI = mapper.readValue(composition, ComponentInstance.class);
			double[] y = characterizer.characterize(cI);
			String serializedY = Arrays.stream(y).mapToObj(d -> d + "").collect(Collectors.joining(" "));
			Map<String, Object> values = new HashMap<>();
			values.put("dataset", dataset);
			values.put("y", serializedY);
			values.put("score", avgScore);
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
					"CREATE TABLE "+newTableName+" (\r\n" + " `id` int(10) NOT NULL AUTO_INCREMENT,\r\n"
							+ " `dataset` TEXT NOT NULL, \r \n"
							+ " `score` double NOT NULL,\r\n" 
							+ " `y` TEXT NOT NULL, \r \n" + " PRIMARY KEY (`id`)\r\n"
							+ ") ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 COLLATE=utf8_bin",
					new ArrayList<>());
		}
	}

	/**
	 * Trains the TreeMiner with all distinct algorithms.
	 * 
	 * @param adapter the database connector
	 * @param loader componentloader used to initialize the meta learner
	 * @param mapper objectmapper that can deserialize component instances
	 * @return the trained tree miner
	 * @throws SQLException
	 * @throws IOException
	 * @throws JsonMappingException
	 * @throws JsonParseException
	 * @throws InterruptedException
	 */
	private static WEKAPipelineCharacterizer minePatterns(SQLAdapter adapter, ComponentLoader loader,
			ObjectMapper mapper)
			throws SQLException, JsonParseException, JsonMappingException, IOException, InterruptedException {
		// selects all distinct pipelines
		ResultSet res = adapter.getResultsOfQuery("SELECT DISTINCT composition from performance_samples");

		List<ComponentInstance> instances = new ArrayList<>();
		
		// converts them to component instances
		while (res.next()) {
			String compositionJson = res.getString(1);
			ComponentInstance ci = mapper.readValue(compositionJson, ComponentInstance.class);
			instances.add(ci);
		}
		// trains the treeminer
		WEKAPipelineCharacterizer metaLearner = new WEKAPipelineCharacterizer(loader.getParamConfigs());
		metaLearner.build(instances);
		return metaLearner;
	}
}
