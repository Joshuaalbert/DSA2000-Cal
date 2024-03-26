Forward modelling is broken up into various functions, that represent different parts of the real total system. The goal is to be able to accurately model each component of the real system within resource constraints. That means that this is a high-performance computing problem almost as much as the real system. Many questions will be answered with forward modelling, and each will require a unique setup. Therefore, the most effective way to do this is to treate the system in a unified approach, and build a library of functions that can be orchestrated across computational resources. 

These are the modular functions:

```haskell
Domain: Gain Models
SimulateIonosphere :: () -> GainModel
SimulateDishDefects :: () -> GainModel

Domain: Predict Visibilities
DFTPredict :: (DiscreteSkyModel, GainModel) -> Visibilties
FFTPredict :: (SkyImage, GainModel) -> Visibilties
RFIPredict :: (EmitterModel, GainModel) -> Visibilities

Domain: Process Visibilties
Flag :: (Visibilities) -> Visibilities
Average :: (Visibilties) -> Visibilities

Domain: Calibration
Calibrate :: (Visibilties, DiscreteSkyModel, GainModel) -> GainModel
Subtract :: (Visibilties, DiscreteSkyModel, GainModel) -> Visibilties

Domain: Imaging
Image :: (Visibilties, GainModel) -> SkyImage
```

We can stitch these together to explore individual questions. Some questions may involve changing forward modelling source code of the above methods, as there is much innovation that must be done to build the the real system. That is why it is important that the forward modelling code is self-contained and consistent. In addition, to simulate on the scale of the DSA2000 the forward modelling itself requires innovation.

Here are some prioritised current questions for forward modelling team:
1. What is the impact of cadencing calibration over a full 10.3min observation?
2. Can we calibrate LWA data, and subtract bright sources and get desired DR?
3. Is the full end-to-end system with realistic systematics going to perform to specifications?
4. What is the effect of moving antennas?

Here are some past questions that forward modelling has answered:
1. What amount of data and averaging is required to calibrate a typical ionosphere to < 2deg RMS phase error over aperature?
2. Can we reach real-time calibration?

Prioritised backlog:
1. Cadenced calibration needs to be implemented.

- [ ] Add as functionality into calibration.

1. Simulating a 10.3 minute observation requires sequential forward modelling utilising high-performance distributed data sharding.

- [ ] Implement XLA data sharding and sequential generation.
- [ ] SimulateIonosphere needs to be carefully reworked to get 10.3 simulation, perhaps using sharding if necessary.
- [ ] Assess what machine(s) will be needed for full simuation.

2. FFTPredict and Imaging require image domain degridding and gridding implementations.

- [ ] IDG implementation in XLA.