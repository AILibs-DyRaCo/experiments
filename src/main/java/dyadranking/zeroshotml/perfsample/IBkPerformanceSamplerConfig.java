package dyadranking.zeroshotml.perfsample;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/perfsampler/ibk_sampler.properties" })
public interface IBkPerformanceSamplerConfig extends IMultiClassClassificationExperimentConfig {

}
