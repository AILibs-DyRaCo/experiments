package dyadranking.zeroshotml.experiments;

import org.aeonbits.owner.Config.Key;
import org.aeonbits.owner.Config.Sources;
import org.aeonbits.owner.Mutable;

@Sources({ "file:./conf/zeroshotexp/curves_evaluator_aux.properties" })
public interface IAuxZeroShotCurvesEvaluatorConfig extends Mutable {
	
	public static final String PARAMS_FOLDER = "paramsfolder";
	
	public static final String PARAMS_FILE_PREFIX = "paramsfile_prefix";
	
	public static final String PARAMS_FILE_SUFFIX = "paramsfile_suffix";
	
	@Key(PARAMS_FOLDER)
	public String getParamsFolder();
	
	@Key(PARAMS_FILE_PREFIX)
	public String getParamsFilePrefix();
	
	@Key(PARAMS_FILE_SUFFIX)
	public String getParamsFileSuffix();
	
}
