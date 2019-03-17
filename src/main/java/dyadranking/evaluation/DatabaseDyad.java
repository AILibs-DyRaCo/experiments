package dyadranking.evaluation;

public class DatabaseDyad {

	private int datasetId;
	
	private String serializedY;
	
	private double score;
	
	private int indexInDB;

	public DatabaseDyad(int datasetId, String serializedY, double score, int indexInDB) {
		super();
		this.datasetId = datasetId;
		this.serializedY = serializedY;
		this.score = score;
		this.indexInDB = indexInDB;
	}

	public int getDatasetId() {
		return datasetId;
	}

	public String getSerializedY() {
		return serializedY;
	}

	public double getScore() {
		return score;
	}

	public int getIndexInDB() {
		return indexInDB;
	}
	
	
}