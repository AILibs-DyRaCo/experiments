package dyadranking.nodeevaluation.timebased;

import java.util.List;

import org.aeonbits.owner.Mutable;
import org.aeonbits.owner.Config.Sources;

@Sources({ "file:conf/timebasedevals/timebased.properties" })
public interface MLPlanTimeBasedExperimenterConfig extends Mutable{

	public static String TIMEOUT_KEY = "timeouts";
	
	public static String DATASET_KEY = "datasets";
	
	public static String SEED_KEY = "seeds";
	
	public static String USE_DYADRANKING = "useDyadRanking";
	
	
	@Key(TIMEOUT_KEY)
	public List<Long> getTimeouts();

	@Key(DATASET_KEY)
	public List<Integer> getOpenMLIds();
	
	@Key(SEED_KEY)
	public List<Integer> getSeeds();
	
	@Key(USE_DYADRANKING)
	public boolean useDyadRanking();
} 
