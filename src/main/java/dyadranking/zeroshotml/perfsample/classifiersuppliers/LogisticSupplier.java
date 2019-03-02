package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.Utils;

public class LogisticSupplier implements IClassifierSupplier {

	@Override
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception {
		String R_ridge_option = "-R " + description.get("R_ridge");
		String M_max_iterations_option = "-M " + description.get("M_max_iterations");
		
		Logistic log = new Logistic();
		String options = 
				R_ridge_option +
				M_max_iterations_option;
		String[] optionsSplit = Utils.splitOptions(options);
		log.setOptions(optionsSplit);
		
		return log;
	}

}
