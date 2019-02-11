package dyadranking.zeroshotml.perfsample;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/perfsampler/j48_sampler.properties" })
public interface IJ48PerformanceSamplerConfig extends IMultiClassClassificationExperimentConfig {

}
