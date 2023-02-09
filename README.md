# ARGA: Abstract Reasoning with Graph Abstractions

This repository contains the implementation for ARGA as described in the AAAI-23 paper [Graphs, Constraints, and Search for the Abstraction and Reasoning Corpus](https://arxiv.org/abs/2210.09880).
Some code were edited for formalism, which caused some small variations in the results as presented in the paper.

To run ARGA on any ARC task, simply call

`
python main.py taskid.json tasktype(training/evaluation)
`

for example, to run the task #ddf7fa4f:

`
python main.py ddf7fa4f.json training
`

the results will be stored under the `solutions` folder and the visualization for the results will be stored under the `images` folder.

Example tasks used for illustration in the paper are as follows:
- recolor task: d2abd087.json
- dynamic recolor task: ddf7fa4f.json
- movement task: 3906de3d.json
- augmentation task: d43fd935.json

## dataset
The ids for the 160 selected tasks can be found under `dataset/subset`.
The full set of the ARC dataset created by Chollet can be found [here](https://github.com/fchollet/ARC) as well as under the `dataset` folder.
