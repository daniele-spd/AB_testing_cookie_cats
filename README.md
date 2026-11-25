# A/B testing on a variant of the game Cookie Cats
Control group (A) is the version of the game with a gate that forces players to wait or make an in-app purchase to progress at level 30, while Treatment (B) moves the gate at level 40.
The primary metrics was retention 7 days after installation (retention 7), but there was also a secondary metric (retention 1).

# Description
The main goals were:
* assess whether the variant improved the retention 7 or not
* study the behavior of inactive (did not login) and passive (did not play) users

*Dataset* = data obtained from the public dataset https://www.kaggle.com/datasets/arpitdw/cokie-cats?resource=download


*Tools* = Python

# Example analysis
## Python
Data understanding, checking the meaning of variables
```python
print("Users with retention 1 = True but no games:")
print(df.loc[(df["retention_1"] == True) &
    (df["sum_gamerounds"] == 0)
]["userid"].count())
print("Users with retention 7 = True but no games:")
print(df.loc[(df["retention_7"] == True) &
    (df["sum_gamerounds"] == 0)
]["userid"].count())
print("Users with retention 7 = True but retention 1 = False:")
print(df.loc[(df["retention_7"] == True) &
    (df["retention_1"] == False)
]["userid"].count())
```

Data Cleaning (not necessary in this case, btw)
```python
df["userid"].duplicated().sum()
```

Checking qualitative balance between groups (Standardized Mean Difference) for number of games
```python
group_A = df[df["version"] == "gate_30"]["sum_gamerounds"]
group_B = df[df["version"] == "gate_40"]["sum_gamerounds"]

mean_A = group_A.mean()
mean_B = group_B.mean()

std_A = group_A.std()
std_B = group_B.std()

pooled_std = np.sqrt(
    (std_A**2 + std_B**2) / 2
)

smd = (mean_A - mean_B) / pooled_std

print(f"SMD: {smd:.4f}")
print(f"Absolute SMD: {abs(smd):.4f}")

if abs(smd) < 0.05:
    print("Excellent balance")
elif abs(smd) < 0.10:
    print("Acceptable balance")
elif abs(smd) < 0.20:
    print("Potential imbalance")
else:
    print("Relevant imbalance")
```





Permutation test (explorative analysis with 1k permutations)
```python
metrics = ["retention_1", "retention_7"]

results = []

for metric in metrics:

    x = df.loc[df["version"] == "gate_30"][metric]
    y = df.loc[df["version"] == "gate_40"][metric]

    res = permutation_test(
        (x, y),
        lambda a, b: np.mean(a) - np.mean(b),
        n_resamples=1_000,
        alternative='two-sided'
    )

    results.append({
        "Metric": metric,
        "Mean gate_30": np.mean(x),
        "Mean gate_40": np.mean(y),
        "Difference": np.mean(x) - np.mean(y),
        "p-value": res.pvalue
    })

results_df = pd.DataFrame(results)


# Holm-Bonferroni correction
reject, p_adjusted, _, _ = multipletests(
    results_df["p-value"],
    alpha=0.05,
    method='holm'
)

results_df["Adjusted p-value (Holm)"] = p_adjusted
results_df["Significant"] = reject

# Formatting
results_df = results_df.round(5)

display(results_df)
```



# Insights
The main results (summarized in "01 - AB testing report.pdf" and "02 - Inactive users report.pdf") include: 
* the variant should not be implemented, since it reduces retention 7 (absolute downlift of -0.59%) with statistical significance
* for retention 1, the Minimum Detectable Effect  is higher than the observed effect, and there is not enough evidence to conclude that groups are statistically different  
* 56% of users becomes inactive the 1st day, and 67% of them becomes inactive between the 2nd and the 7th day.
* 93% of users who were inactive the first day, kept being inactive.
* 4% of users are passive, and 98% of them was inactive after 1 day, becoming 99% after 7 days.

# How to
Useful links for data source and data interpretation are in "documentation.xlsx".

CSV data are in the folder "raw_data". 

Jupyter Notebooks "01 - AB testing report.ipynb" and "02 - Inactive users report.ipynb" can run the tests again.

An overview of the project and the corresponding results is in "01 - AB testing report.pdf" and "02 - Inactive users report.pdf".

## Requirements
Python 3.12.12

## Installation
* Clone the repository: 
    * git clone https://github.com/daniele-spd/AB_testing_cookie_cats.git
* Navigate to the project directory: 
    * cd REPOSITORY
* (Optional but recommended) Create a virtual environment: 
    * python -m venv venv
* Activate the environment: 
    * Windows: venv\Scripts\activate 
    * Linux / macOS: source venv/bin/activate
* Install the dependencies: 
    * pip install -r requirements.txt