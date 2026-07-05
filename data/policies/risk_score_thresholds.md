# Risk Score Thresholds

## Score Bands

The fraud model produces a probability score between 0.0 and 1.0 for each
case. Scores below 0.30 are considered low risk and are eligible for
automatic approval provided no sanctions hit is present. Scores from 0.30
up to and including 0.70 fall in the gray zone and require manual review
by a compliance analyst. Scores above 0.70 are high risk and must be
escalated regardless of any other signal.

## Score Recalculation

If new transaction data arrives for a case that has already been scored,
the fraud score must be recalculated before a final decision is recorded.
A stale score older than 24 hours may not be used to approve a case.

## Interaction with Other Signals

A sanctions hit or a failed identity verification always overrides the
risk score band: even a low fraud score does not permit automatic
approval if either of those conditions is present.
