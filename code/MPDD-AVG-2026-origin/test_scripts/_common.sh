#!/usr/bin/env bash

dataset_root_for_split() {
  local split_name="$1"
  printf 'MPDD-AVG2026/MPDD-AVG2026-test/%s\n' "$split_name"
}

split_csv_for_split() {
  local split_name="$1"
  printf 'MPDD-AVG2026/MPDD-AVG2026-test/%s/split_labels_test.csv\n' "$split_name"
}

resolve_personality_npy() {
  local split_name="$1"
  printf 'MPDD-AVG2026/MPDD-AVG2026-trainval/%s/descriptions_embeddings_with_ids.npy\n' "$split_name"
}
