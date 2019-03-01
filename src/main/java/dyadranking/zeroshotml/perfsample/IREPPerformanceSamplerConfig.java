package dyadranking.zeroshotml.perfsample;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/perfsampler/rep_sampler.properties" })
public interface IREPPerformanceSamplerConfig extends IMultiClassClassificationExperimentConfig {

}
