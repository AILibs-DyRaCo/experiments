package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.Utils;

public class REPSupplier implements IClassifierSupplier {

	@Override
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception {
		String M_num_instances_option = " -M " + description.get("M_num_instances");
		String V_min_variance_split_option = "-V " + description.get("V_min_variance_split");
		String L_max_depth_option = "-L " + description.get("L_max_depth");
		
		REPTree rep = new REPTree();
		String options = 
				M_num_instances_option +
				V_min_variance_split_option +
				L_max_depth_option;
		String[] optionsSplit = Utils.splitOptions(options);
		rep.setOptions(optionsSplit);
		
		return rep;
	}

}
