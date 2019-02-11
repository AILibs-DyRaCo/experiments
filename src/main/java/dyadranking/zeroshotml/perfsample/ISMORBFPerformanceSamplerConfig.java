package dyadranking.zeroshotml.perfsample;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;
@Sources({ "file:./conf/perfsampler/smo_rbf_sampler.properties" })
public interface ISMORBFPerformanceSamplerConfig extends IMultiClassClassificationExperimentConfig {

}
