package dyadranking.zeroshotml.perfsample;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/perfsampler/rt_sampler.properties" })
public interface IRTPerformanceSamplerConfig extends IMultiClassClassificationExperimentConfig {

}
