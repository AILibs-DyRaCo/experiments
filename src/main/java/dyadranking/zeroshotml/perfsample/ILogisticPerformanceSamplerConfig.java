package dyadranking.zeroshotml.perfsample;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/perfsampler/logistic_sampler.properties" })
public interface ILogisticPerformanceSamplerConfig extends IMultiClassClassificationExperimentConfig {

}
