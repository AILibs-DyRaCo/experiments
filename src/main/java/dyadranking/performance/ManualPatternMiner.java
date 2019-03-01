package dyadranking.performance;

import java.io.File;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.isys.linearalgebra.Vector;
import dyadranking.sql.SQLUtils;
import hasco.model.CategoricalParameterDomain;
import hasco.model.Component;
import hasco.model.ComponentInstance;
import hasco.model.Parameter;
import hasco.serialization.ComponentLoader;
import jaicore.basic.SQLAdapter;

public class ManualPatternMiner {

	private static final Logger logger = LoggerFactory.getLogger(ManualPatternMiner.class);

	/*
	 * Maps the name of a component to a map that maps the name of the hyper
	 * parameter to its index in the dyad vector.
	 */
	Map<String, Map<String, Integer>> componentNameToParameterDyadIndex = new HashMap<>();

	Map<String, Integer> componentNameToDyadIndex = new HashMap<>();

	int patternCount;

	public ManualPatternMiner(Collection<Component> collection) {
		int counter = 0;
		logger.debug("Got {} components as input.", collection.size());
		for (Component component : collection) {
			logger.debug("Inserting {} at position {}", component.getName(), counter);
			componentNameToDyadIndex.put(component.getName(), counter++);
			Map<String, Integer> parameterIndices = new HashMap<>();
			logger.debug("{} has {} parameters.", component.getName(), component.getParameters().size());
			for (Parameter param : component.getParameters()) {
				if (param.isNumeric()) {
					parameterIndices.put(param.getName(), counter++);
				} else if (param.isCategorical()) {
					parameterIndices.put(param.getName(), counter);
					CategoricalParameterDomain domain = (CategoricalParameterDomain) param.getDefaultDomain();
					counter += domain.getValues().length;
				}
			}
			componentNameToParameterDyadIndex.put(component.getName(), parameterIndices);
		}
		System.out.println("Found " + counter);
		patternCount = counter;
	}

	/**
	 * Recursively, resolves the components.
	 * 
	 * @param cI
	 * @param input
	 * @return
	 */
	public double[] characterize(ComponentInstance cI, Vector patterns) {
		// Vector patterns = new DenseDoubleVector(patternCount);
		// first: get the encapsulated component
		Component c = cI.getComponent();
		String componentName = c.getName();
		// set the used algorithm to '1'
		int index = componentNameToDyadIndex.get(componentName);
		patterns.setValue(index, 1.0d);
		// now resolve the parameters
		Map<String, Integer> parameterIndices = componentNameToParameterDyadIndex.get(componentName);
		// assumption: the values is always set in the parameters vector
		for (Parameter param : c.getParameters()) {
			String parameterName = param.getName();
			int parameterIndex = 0;
			parameterIndex = parameterIndices.get(parameterName);
			if (param.isNumeric()) {
				double value = Double.parseDouble(cI.getParameterValue(param));
				patterns.setValue(parameterIndex, value);
			} else if (param.isCategorical()) {
				// the parameters are one-hot-encoded, where the parameterIndex specifies the
				// one hot index for the first categorical parameter, parameterIndex+1 is the
				// one-hot index for the second parameter etc.
				String parameterValue = cI.getParameterValue(param);
				CategoricalParameterDomain domain = (CategoricalParameterDomain) param.getDefaultDomain();
				for (int i = 0; i < domain.getValues().length; i++) {
					if (domain.getValues()[i].equals(parameterValue)) {
						patterns.setValue(parameterIndex + i, 1);
					} else {
						patterns.setValue(parameterIndex + i, 0);
					}
				}

			}
		}
		// recursively resolve the patterns for the requiredInterfaces
		for (ComponentInstance requiredInterface : cI.getSatisfactionOfRequiredInterfaces().values()) {
			characterize(requiredInterface, patterns);
		}
		return patterns.asArray();
	}
}
