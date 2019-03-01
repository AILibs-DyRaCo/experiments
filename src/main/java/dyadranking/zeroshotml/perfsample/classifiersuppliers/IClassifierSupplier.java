package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.core.Instances;

public interface IClassifierSupplier {
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception;
}
