# Experimentation, A/B Testing and Causal Inference

التجارب واختبارات A/B والاستدلال السببي

| | |
|---|---|
| Provider | SDAIA Academy |
| Program code | SDA-DSC-213 |
| Duration | 4 days, 20 hours (5 hours per day) |
| Dates | Sunday 6 to Wednesday 9 September 2026 |
| Instructor | Abrar Alotaibi |
| Tools | Python, pandas, statsmodels, scipy, Jupyter |

## What the program covers

The program trains participants to answer "what actually causes the effect" with data, through experiments and causal inference. Participants design randomized experiments with correct randomization units and guardrail metrics; compute statistical power, sample sizes and minimum detectable effects; analyze experiment results with correct variance estimation and multiple-testing control while avoiding pitfalls such as peeking; apply quasi-experimental methods (matching, difference-in-differences, instrumental variables) when experiments are not possible; and assess causal claims from observational data using causal graphs (DAGs). The program ends with an experiment design and analysis project.

## Schedule

| Day | Date | Theme | Lab |
|---|---|---|---|
| 1 | Sun 6 Sep | Causal thinking and potential outcomes | Lab 1 |
| 2 | Mon 7 Sep | Randomized experiments and A/B testing end to end | Lab 2 |
| 3 | Tue 8 Sep | Observational and quasi-experimental methods | Lab 3 |
| 4 | Wed 9 Sep | Uncertainty, bias, cases and the capstone | Lab 4 |

The entrance exam is taken at the start of Day 1 and the exit exam at the end of Day 4 (links shared in class).

## Folders

- `day1/` slides for the whole workshop and the Day 1 revision handout. Handouts for Days 2 to 4 are added at the end of each day.
- `labs/starter/` the four lab notebooks with TODO cells. Open them in Jupyter or Colab and work from `labs/`.
- `labs/data/` the synthetic Injaz datasets the labs use.
- `labs/causal_utils.py` helper functions shared by all labs.
- `labs/solution/` posted after the workshop, together with `data/ground_truth.json`.

## Setup

```
cd labs
pip install -r requirements.txt
python check_env.py
jupyter lab
```

The notebooks expect to be run from inside `labs/starter/` (they import `causal_utils` from the parent folder and read data from `../data`).

## Labs

| Lab | Notebook | Day |
|---|---|---|
| 1 | lab1_potential_outcomes | 1 |
| 2 | lab2_ab_test_end_to_end | 2 |
| 3 | lab3_four_estimators | 3 |
| 4 | lab4_capstone | 4 |
