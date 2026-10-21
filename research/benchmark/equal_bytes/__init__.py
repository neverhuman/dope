"""Equal-byte transfer study: a DOPE program versus shipping real rows.

The study asks the question behind Transfer Description Length directly. For a
byte budget B, which representation of a table gives downstream learners the
most of the real data's predictive gain: a compiled DOPE program of at most B
bytes, or the largest set of real rows that fits in B bytes under the best of
several lossless codecs? See README.md in this directory.
"""
