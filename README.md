# Monte Carlo Portfolio Optimization (Long Only & Long/Short Strategies)

This repository contains the implementation and report for a Monte Carlo–based portfolio optimization assignment. We explore long-only and long‑short portfolio strategies across three assets, maximizing expected return under risk constraints using stochastic programming.

## Repository Contents

- `Monte_Carlo_In_Python.py`
- `Portfolio_Optimization_Report_with_Monte_Carlo_Simulation.pdf` — Technical research report
- `portfolio_optimization_monte_carlo_side_by_side.pdf` — Efficient frontier visualization
- `README.md` — This documentation file

---

## User Instructions

### How to Run Python Script
1. Run the script:

   ```bash
   python Monte_Carlo_In_Python.py
   ```

2. Output:
   - Side-by-side efficient frontier plot (`.pdf`)
   - Summary statistics printed in terminal or interactive window

---

## Testing Instructions
- Confirm output plot (`portfolio_optimization_monte_carlo_side_by_side.pdf`) is generated
- Modify asset return or risk parameters and rerun to test flexibility
- Compare behavior of `shorts_ok=True` vs `shorts_ok=False` in simulations

---

## Design Summary
1. **Define Inputs**: Expected returns, volatilities, correlation matrix  
2. **Simulate Returns**: Generate synthetic data using multivariate normal distribution  
3. **Monte Carlo Optimization**: Simulate 10,000 random portfolios  
4. **Compare Strategies**: With and without short selling  
5. **Visualize Efficient Frontier**: Show risk-return tradeoffs in a side-by-side plot  
6. **Summarize Statistics**: Print descriptive metrics for both strategies  

---

## Deliverables Checklist
- [x] Python script or notebook  
- [x] Exported efficient frontier PDF  
- [x] PDF report  
- [x] README.md with full documentation  
- [x] Public GitHub repository (link submitted ends in `.git`)  

---

## Use of AI Assistants
AI tools, including **ChatGPT (OpenAI)**, were used to complete this assignment in the following purposes:
- Clarifying concepts like stochastic programming and efficient frontier  
- Troubleshooting scripts I initially wrote  
- Assisting in drafting Markdown formatting  
- Proofreading my PDF research output for clarity and readability  

---
