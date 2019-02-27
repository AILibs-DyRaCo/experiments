package dyadranking.zeroshotml.experiments;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import dyadranking.sql.SQLUtils;
import jaicore.basic.SQLAdapter;

public class ZeroShotCurvesCollector {
	
	private static final String[] CURVE_NAMES = new String[] { "j48_default_183_1" };
	
	private static final int NUM_ITERATIONS = 200;
	
	private static final String OUTPUT_PATH = "datasets/zeroshot/eval";
	
	private static final String TABLE = "`zeroshot_curves_evaluation`";
	
	private static final String INPUT_FILE_ENDING = ".rawpars";
	
	private static final String OUTPUT_FILE_ENDING = ".perfs";
	
	public static void main(String[] args) throws SQLException, IOException {
		SQLAdapter adapter = SQLUtils.sqlAdapterFromArgs(args);
		
		for (String curveName : CURVE_NAMES) {
			ResultSet res = adapter.getResultsOfQuery(
					"SELECT iteration_index, percent_correct FROM " + TABLE + 
					"WHERE parameter_file = \"" + curveName + INPUT_FILE_ENDING + "\"");
			res.first();
			double[] perfsArr = new double[NUM_ITERATIONS];
			do {
				int iterationIndex = res.getInt("iteration_index");
				double pctCorrect = res.getDouble("percent_correct");
				perfsArr[iterationIndex] = pctCorrect;
			} while(res.next());

			File outputFile = new File(OUTPUT_PATH + File.separator + curveName + OUTPUT_FILE_ENDING);
			if (!outputFile.exists()) {
				outputFile.createNewFile();
			}
			PrintWriter outputStream = null;
			outputStream = new PrintWriter(outputFile);
			for (double perf : perfsArr) {
				outputStream.println(perf);
			}
			outputStream.close();
		}
	}

}
