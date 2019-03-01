package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;

public class RFSupplier implements IClassifierSupplier {

	@Override
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception {
		String I_iteration_option = " -I " + description.get("I_iterations");
		String M_num_instances_option = " -M " + description.get("M_num_instances");
		String depth_option = " -depth " + description.get("depth");				
		int K_num_attributes = (int) Math.ceil(data.numAttributes() * Double.parseDouble(description.get("K_fraction_attributes")));
		String K_option = " -K " + K_num_attributes;
		
		RandomForest rf = new RandomForest();
		String options = 
				I_iteration_option +
				K_option +
				M_num_instances_option +
				depth_option;
		String[] optionsSplit = Utils.splitOptions(options);
		rf.setOptions(optionsSplit);
		
		return rf;
	}

}
