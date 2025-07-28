import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetic style for seaborn
sns.set(style="whitegrid")

# -----------------------------
# Step 1: Define the parameters
# -----------------------------
np.random.seed(1111)
sample_size = 10000

# Mean returns for assets A, B, C, D
mean_returns = np.array([0.02, 0.07, 0.15, 0.20])

# Standard deviations
std_devs = np.array([0.05, 0.12, 0.17, 0.25])

# Correlation matrix
corr_matrix = np.array([
    [1,   0.3, 0.3, 0.3],
    [0.3, 1,   0.6, 0.6],
    [0.3, 0.6, 1,   0.6],
    [0.3, 0.6, 0.6, 1]
])

# Covariance matrix
cov_matrix = np.diag(std_devs) @ corr_matrix @ np.diag(std_devs)

# ----------------------------------
# Step 2: Generate synthetic returns
# ----------------------------------
from numpy.random import multivariate_normal

returns_data = multivariate_normal(mean_returns, cov_matrix, size=sample_size)
returns_df = pd.DataFrame(returns_data, columns=["A", "B", "C", "D"])

# ----------------------------------
# Step 3: Define weight generators
# ----------------------------------
def make_weights(shorts_ok=True):
    if shorts_ok:
        w = np.random.uniform(-1, 1, 3)
        w4 = 1 - np.sum(w)
        return np.append(w, w4)
    else:
        w = np.random.uniform(0, 1, 4)
        return w / np.sum(w)

# ----------------------------------
# Step 4: Run simulation
# ----------------------------------
def run_simulation(shorts_ok=True, label="Shorts OK"):
    np.random.seed(9999)
    all_results = []
    for _ in range(sample_size):
        weights = make_weights(shorts_ok)
        port_returns = returns_data @ weights
        mean_ret = np.mean(port_returns)
        std_ret = np.sqrt(weights.T @ cov_matrix @ weights)
        has_shorts = np.any(weights < 0)
        all_results.append({
            "w1": weights[0], "w2": weights[1],
            "w3": weights[2], "w4": weights[3],
            "returnMean": mean_ret, "returnSD": std_ret,
            "Positions": "Has Short(s)" if has_shorts else "No Shorts",
            "ShortsOK": label
        })
    return pd.DataFrame(all_results)

df_short = run_simulation(shorts_ok=True, label="Shorts OK")
df_long = run_simulation(shorts_ok=False, label="Long Positions Only")

# Combine results
results = pd.concat([df_short, df_long], ignore_index=True)

# ----------------------------------
# Step 5: Plotting the results
# ----------------------------------
# Step 5: Plotting the results (top and bottom layout)
# Step 5: Plotting the results (side-by-side layout with same x and y scales)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True, sharex=True)

# Subset data
shorts_df = results[results["ShortsOK"] == "Shorts OK"]
longs_df = results[results["ShortsOK"] == "Long Positions Only"]

# Determine shared axis limits
x_min, x_max = results["returnSD"].min(), results["returnSD"].max()
y_min, y_max = results["returnMean"].min(), results["returnMean"].max()

# Plot 1: Shorts OK
sns.scatterplot(
    data=shorts_df,
    x="returnSD", y="returnMean",
    hue="Positions",
    palette={"Has Short(s)": "red", "No Shorts": "darkblue"},
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title("Portfolios with Shorts")
axes[0].set_xlabel("Risk (Standard Deviation)")
axes[0].set_ylabel("Return (Mean)")
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
axes[0].legend()

# Plot 2: Long Only
sns.scatterplot(
    data=longs_df,
    x="returnSD", y="returnMean",
    hue="Positions",
    palette={"Has Short(s)": "red", "No Shorts": "darkblue"},
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title("Portfolios without Shorts")
axes[1].set_xlabel("Risk (Standard Deviation)")
axes[1].set_ylabel("")  # y-label only on left
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[1].legend()

# Layout & export
plt.suptitle("Monte Carlo Portfolio Optimization", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("C:\\Users\\Genki\\Documents\\Homework\\Financial Machine Learning\\HW2\\Efficient_Frontier_Side_by_Side_Plot.pdf")
plt.show()

# ----------------------------------
# Step 6: Save all simulated portfolios
# ----------------------------------
results.to_csv("C:\\Users\\Genki\\Documents\\Homework\\Financial Machine Learning\\HW2\\monte_carlo_all_portfolios.csv", index=False)
print("✅ All portfolio simulations saved to monte_carlo_all_portfolios.csv")

# ----------------------------------
# Step 7: Save summary statistics
# ----------------------------------
summary_rows = []

def summarize_portfolio(df, label):
    summary = df[["returnMean", "returnSD"]].describe()
    neg_return_pct = (df["returnMean"] < 0).mean() * 100
    corr = df["returnMean"].corr(df["returnSD"])
    
    print(f"\n=== Portfolio Summary: {label} ===")
    print(summary)
    print(f"Proportion with negative returns: {neg_return_pct:.2f}%")
    print(f"Correlation between return and risk: {corr:.2f}")
    
    # Append for CSV
    summary_rows.append({
        "Strategy": label,
        "Return Mean (avg)": summary.loc["mean", "returnMean"],
        "Return SD (avg)": summary.loc["mean", "returnSD"],
        "Return Mean (min)": summary.loc["min", "returnMean"],
        "Return SD (max)": summary.loc["max", "returnSD"],
        "Negative Return %": neg_return_pct,
        "Corr(Return, Risk)": corr
    })

# Run summaries and save
summarize_portfolio(df_short, "Shorts OK")
summarize_portfolio(df_long, "Long Positions Only")

pd.DataFrame(summary_rows).to_csv("C:\\Users\\Genki\\Documents\\Homework\\Financial Machine Learning\\HW2\\monte_carlo_summary_stats.csv", index=False)
print("✅ Summary statistics saved to monte_carlo_summary_stats.csv")

