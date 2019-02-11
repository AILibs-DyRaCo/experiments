package dyadranking.zeroshotml.perfsample;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/perfsampler/mlp_sampler.properties" })
public interface IMLPPerformanceSamplerConfig extends IMultiClassClassificationExperimentConfig {

}
