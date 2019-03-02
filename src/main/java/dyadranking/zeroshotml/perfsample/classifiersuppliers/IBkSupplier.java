package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.Utils;

public class IBkSupplier implements IClassifierSupplier {

	@Override
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception {
		String K_num_neighbors_option = "-K " + description.get("K_num_neighbors");
		
		IBk ibk = new IBk();
		String options = 
				K_num_neighbors_option;
		String[] optionsSplit = Utils.splitOptions(options);
		ibk.setOptions(optionsSplit);
		
		return ibk;
	}

}
