package dyadranking.performance.mccvevaluation;

public enum APPROACH {
	ONE_NN_BASELINE("1NN_BASELINE"), AVERAGE_RANK_BASELINE("AVERAGE_RANK_BASELINE"), DYADRANKING("DYADRANKING");
	String tableName;

	APPROACH(String tableName) {
		this.tableName = tableName;
	}

	@Override
	public String toString() {
		return tableName;
	}
}
