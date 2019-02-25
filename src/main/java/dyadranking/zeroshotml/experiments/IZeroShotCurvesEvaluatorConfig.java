package dyadranking.zeroshotml.experiments;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/zeroshotexp/curves_evaluator.properties" })
public interface IZeroShotCurvesEvaluatorConfig extends IMultiClassClassificationExperimentConfig {

}
