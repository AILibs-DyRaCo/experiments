package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.Utils;

public class RTSupplier implements IClassifierSupplier {

	@Override
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception {
		int K_num_attributes = (int) Math.ceil(data.numAttributes() * Double.parseDouble(description.get("K_fraction_attributes")));
		String K_option = " -K " + K_num_attributes;
		String M_num_instances_option = " -M " + description.get("M_num_instances");
		String V_min_variance_split_option = "-V " + description.get("V_min_variance_split");
		String depth_option = " -depth " + description.get("depth");				
		
		RandomTree rt = new RandomTree();
		String options = 
				K_option +
				M_num_instances_option +
				V_min_variance_split_option +
				depth_option;
		String[] optionsSplit = Utils.splitOptions(options);
		rt.setOptions(optionsSplit);
		
		return rt;
	}

}
