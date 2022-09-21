# Word-Based-SMT
Implementation of Word based alignment translation models

SMT models for word based translations to aligns words automatically.

1. Dice Algorithm
2. IBM Model 1 + EM Algorithm

Usage:

There are four python programs here (-h for usage):

1. ./align aligns words using the Dice Algorithm.
2. ./ibm_model_1 aligns words using IBM Model 1 + EM Algorithm.
3. ./check-alignments checks that the entire dataset is aligned, and that there are no out-of-bounds alignment points.
4. ./score-alignments computes alignment error rate.