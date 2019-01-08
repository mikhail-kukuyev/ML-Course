#!/usr/bin/env bash
./learner.py train model
./checker.py ./applicator.py model train
