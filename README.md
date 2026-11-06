# Vacuum World — Q Learning

## Overview

Implementation of the AIMA vacuum world (4x4 grid) using Q Learning

## Environment

4x4 grid, initial state:

start 0 0 0
  0   X 0 0
  0   0 0 0
  0  -1 0 10

- `start` = initial robot position (0,0)
- `X` = obstacle (impassable)
- `-1` = penalty cell
- `10` = reward cell
- `0` = plain cell

Actions: `UP`, `DOWN`, `LEFT`, `RIGHT`, `SUCK`, `NO-OP`.

## Project Structure

.
├── README.md
├── spec.md # Specification file handed to the LLM
├── requirements.txt
├── bootstrap.ipynb
├── vacuum_world_lab4.ipynb # Builds off Lab4, Lab5 Testing at the bottom of the notebook
├── trainVacuum.py #training the qlearning agent
├── qLearning_Agent.py #implements Q Learning strategy
├── video_links.txt #links to YouTube videos
└── /aima # AIMA repository

## How to Run

Use vacuum_world_lab4.ipynb for the code to run tests in /Test
