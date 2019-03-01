package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.Utils;

public class MLPSupplier implements IClassifierSupplier {

	@Override
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception {
		String L_learning_rate_option = "-L " + Math.pow(10, Double.parseDouble(description.get("L_learning_rate_exp")));
		String M_momentum_option = " -M " + Math.pow(10, Double.parseDouble(description.get("M_momentum_exp")));
		String N_num_epochs_option = " -N " + description.get("N_num_epochs");
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		String options = 
				L_learning_rate_option +
				M_momentum_option +
				N_num_epochs_option;
		String[] optionsSplit = Utils.splitOptions(options);
		mlp.setOptions(optionsSplit);
		
		return mlp;
	}

}
