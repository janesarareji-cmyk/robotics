# Module 3 — Transformation Engine

## Purpose
Convert DH parameters into homogeneous transformation matrices.

## Input
One DH row:
- theta
- d
- a
- alpha

## Output
One 4x4 matrix.

## Main interfaces
`dh_to_transform(theta, d, a, alpha)`
`build_transform_chain(dh_table)`

## Calculation
For every joint:
T(i-1,i) = f(theta_i, d_i, a_i, alpha_i)

The chain is accumulated:
T0,2 = T0,1 × T1,2
T0,3 = T0,2 × T2,3

## Teaching support
Retain:
- each local transform
- each cumulative transform
- frame index

## Tests
- Zero-parameter identity case
- Known single-link case
- Known 90° case
- Multi-link multiplication
- Matrix dimensions
