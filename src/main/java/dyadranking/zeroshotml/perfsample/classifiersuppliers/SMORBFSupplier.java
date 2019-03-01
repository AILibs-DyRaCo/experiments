package dyadranking.zeroshotml.perfsample.classifiersuppliers;

import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.Utils;

public class SMORBFSupplier implements IClassifierSupplier {

	@Override
	public Classifier setupClassifier(Map<String, String> description, Instances data) throws Exception {
		String C_complexity_const_option = "-C " + Math.pow(10, Double.parseDouble(description.get("C_complexity_const_exp")));
		String L_tolerance_option = " -L " + Math.pow(10, Double.parseDouble(description.get("L_tolerance_exp")));
		String RBF_gamma_option =" -K \"weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G "  
								+ Math.pow(10, Double.parseDouble(description.get("RBF_gamma_exp"))) + "\"";

		SMO smo = new SMO();
		String options = 
				C_complexity_const_option +
				L_tolerance_option +
				RBF_gamma_option;
		String[] optionsSplit = Utils.splitOptions(options);
		smo.setOptions(optionsSplit);
		
		return smo;
	}

}
