#!/usr/bin/env bash

declare -a links=(
  "schnorb_hamiltonian_water.tgz"
  "schnorb_hamiltonian_uracil.tgz"
  "schnorb_hamiltonian_malondialdehyde.tgz"
  "schnorb_hamiltonian_ethanol_hf.tgz"
  "schnorb_hamiltonian_ethanol_dft.tgz"
)

for l in "${links[@]}"; do
  echo "Working with $l"
  wget http://quantum-machine.org/data/schnorb_hamiltonian/$l
  tar -xvf $l || rm $l
done