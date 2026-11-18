# What TriCSI does

A person changes wireless paths between transmitters and a receiver. ESP32-S3
boards record channel state information across subcarriers. TriCSI predicts
stationary occupancy in a 3×3 grid, plus EMPTY; it does not reconstruct body pose.

1. Collect one CSV per zone/orientation trial plus separate Empty trials.
2. Audit packet fields and trial identities; preserve original recording bytes.
3. Construct windows inside whole trials. All windows from a trial stay together
   in every split.
4. The two-link model subtracts an Empty reference from selected inputs, fits
   normalization on real source data, encodes each link with a CNN, and fuses
   embeddings. Occupancy is predicted separately from the occupied zone.
5. Source occupied embeddings form zone prototypes. Window probabilities are
   averaged within each scored trial.
6. Report trial accuracy separately per session, including calibration, exclusions
   and the checkpoint-selection rule.

The three-link pipeline extracts features and compares all seven transmitter
subsets using 3TX_S1/S2. It freezes the selected and predeclared ablation models
before the original evaluation on 3TX_S3.

The original two-link checkpoint scored 38/40 on S8 but 21/40 on S9. Current
augmentation experiments investigate this transfer gap using all eligible source
recordings. Generated perturbations are not validated physical simulations.

Packet = one measured row. Window = a segment of one trial. Trial = one labeled
recording. Session = a recording setup/time. Repeated model seeds are not new
recording days. S8/S9 now inform extension development, so selected new scores on
them cannot establish independent generalization.
