package dyadranking.zeroshotml.perfsample;

import org.aeonbits.owner.Config.Sources;

import jaicore.ml.experiments.IMultiClassClassificationExperimentConfig;

@Sources({ "file:./conf/perfsampler/smopk_sampler.properties" })
public interface ISMOPKPerformanceSamplerConfig extends IMultiClassClassificationExperimentConfig {

}
