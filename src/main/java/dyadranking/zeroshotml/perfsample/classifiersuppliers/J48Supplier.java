package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

public class J48Supplier implements IClassifierSupplier {

	@Override
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception {
		String C_pruning_confidence_option = "-C " + description.get("C_pruning_confidence");
		String M_min_inst_option = " -M " + description.get("M_min_inst");

		J48 j48 = new J48();
		String options = 
				C_pruning_confidence_option +
				M_min_inst_option;
		String[] optionsSplit = Utils.splitOptions(options);
		j48.setOptions(optionsSplit);
		
		return j48;
	}

}
