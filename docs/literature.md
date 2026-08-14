# What the literature says about this method

firesite was built from first principles and then checked against the published
work. This page records what held up, what was already known, and where the tool
is behind the state of the art. Written after the fact on purpose: the checks
below were not used to design the tool, so agreement is evidence rather than
circular reasoning.

## Validated

**Combining fire likelihood with siting optimization is an established method.**
Palaiologou and co-authors optimize watchtower placement in Chalkidiki, Greece by
combining simulated burn probability, topography and *accessibility to candidate
locations*, then solving for visibility coverage with exact programming over 654
candidate positions across 151,890 ha.<sup>[1]</sup> That is the same pipeline
shape firesite uses. The difference is the fire-likelihood input: they simulate
burn probability, firesite measures observed recurrence.

**Access is a formal criterion, not a compromise.** The same study carries
accessibility to candidate sites as an explicit term in the optimization. The
advice to evaluate the roof you can actually mount on, rather than chase a
theoretical optimum, matches the published treatment.

**Siting is usually done without tools, and that is a recognized problem.**
Nel and co-authors are blunt about it: tower sites "have been identified by
foresters and locals with intimate knowledge of the terrain and without the aid
of computational optimisation tools", and moving into new territory without local
knowledge makes the process "cumbersome and daunting".<sup>[2]</sup> They analyse
165 existing tower sites across South Africa, Canada and the USA.

**A 15 km working radius is a reasonable default.** Kücük and co-authors run
their visibility analysis for Turkish fire lookout towers as a 360° sweep at an
18 km radius.<sup>[3]</sup> firesite defaults to 15 km, in the same range.

**Resolution against distance is the governing constraint.** Barmpoutis and
co-authors, in the most cited review of the field, describe terrestrial optical
systems as achieving resolution "depending on camera resolution and distance
between the camera and the event".<sup>[4]</sup> That is the premise behind
`pixels_on_target`.

**False alarms are the practical failure mode, and temporal features are the
mitigation.** The same review notes that colour-based models suffer "high false
alarm rates, since single-colour information is insufficient", and that the
methods which work combine colour with motion information across frames. This
supports treating temporal confirmation, not a better single-frame model, as the
thing that makes a deployment usable.

## Where firesite is behind the state of the art

**No viewshed, and that is the central technique in the literature.** Every
serious siting study is built on a digital elevation model: line-of-sight
computation, terrain masking, and in Nel's case geomorphon landform
classification to shortlist candidates before optimizing.<sup>[2][3]</sup>
firesite ignores terrain entirely, so a position it ranks highly may sit behind a
ridge. This is the single largest gap and the most useful contribution anyone
could make.

**The site search is naive.** The published framing is the covering location
problem, solved with integer linear programming or metaheuristics, and it handles
*several* cameras whose coverage complements each other.<sup>[2][5]</sup>
firesite brute-forces a grid and reports the best single position. Choosing the
best pair of positions is not the same as taking the two best individual ones,
and firesite currently cannot express that.

**Observed history is a proxy for burn probability.** Simulated burn probability
accounts for fuel, weather and topography, including ground that has not burned
yet but will. Observed detections carry the opposite bias: satellites miss the
small fires, so the history skews toward large events.

## Not backed by a citation

The threshold of **8 pixels across a plume** for reliable detection is an
engineering heuristic, not a value taken from a paper. Searches did not surface a
study establishing a minimum pixel count for smoke detection at range. The
relationship it encodes is sound and the reviews support the underlying physics,
but the specific number should be treated as a planning figure to be calibrated
against a real deployment, not as a published constant. If you know of work that
pins it down, please open an issue.

The **30 m incipient plume width** is likewise a working assumption.

## References

1. Palaiologou, P. et al. (2022). Territorial Resilience Through Visibility
   Analysis for Immediate Detection of Wildfires Integrating Fire Susceptibility,
   Geographical Features, and Optimization Methods. *International Journal of
   Disaster Risk Science*. https://doi.org/10.1007/s13753-022-00433-2
2. Nel, A. et al. (2021). Analysis and Exploitation of Landforms for Improved
   Optimisation of Camera-Based Wildfire Detection Systems. *Fire Technology*.
   https://doi.org/10.1007/s10694-021-01120-2
3. Kücük, Ö. et al. (2020). Visibility analysis of fire lookout towers protecting
   the Mediterranean forest ecosystems in Turkey. *Šumarski list*.
   https://doi.org/10.31298/sl.144.7-8.5
4. Barmpoutis, P. et al. (2020). A Review on Early Forest Fire Detection Systems
   Using Optical Remote Sensing. *Sensors*, 20(22), 6442.
   https://doi.org/10.3390/s20226442
5. Nel, A. et al. (2020). Reduced Target-Resolution Strategy for Rapid
   Multi-Observer Site Location Optimisation. *IEEE Access*.
   https://doi.org/10.1109/access.2020.3037025
6. Bouguettaya, A. et al. (2022). Early Wildfire Detection Technologies in
   Practice: A Review. *Sustainability*, 14(19), 12270.
   https://doi.org/10.3390/su141912270
