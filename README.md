## Goal Planner - Investment Strategy Website

A simple, static website that helps users plan investments to achieve goals like buying a first car or saving for a vacation. It estimates the required monthly investment using assumed returns and inflation, and visualizes progress over time.

### Features
- Goal-based planning (target amount, timeframe, starting savings)
- Choice of contribution frequency (monthly/weekly)
- Assumptions for expected annual return and inflation
- Breakdown of required contributions vs. expected portfolio growth
- Interactive chart of expected value over time
- Works entirely in the browser (no backend)

### Quick Start
1. Open `index.html` in any modern browser.
2. Fill in your goal details and assumptions.
3. Click "Calculate Plan" to see recommended contributions and the growth chart.

### Detailed Usage
1. Open the website
   - Double-click `index.html` or use a static server.
   - Optional: run a local server for better file access restrictions:
     - Python 3: `python3 -m http.server 8080` then visit `http://localhost:8080`.
2. Enter goal details
   - Goal name: e.g., "First Car" or "Hawaii Trip".
   - Target amount: how much you need in future currency.
   - Time horizon: years until the goal.
   - Current savings: what you have today toward this goal.
3. Set assumptions
   - Expected annual return: average % return of your investments.
   - Annual inflation: to adjust target to future value (optional; set to 0 to ignore).
   - Contribution frequency: monthly (default) or weekly.
4. Calculate
   - Click "Calculate Plan".
   - The app will:
     - Inflate the target amount to the goal date using the inflation rate.
     - Compute the required periodic contribution to reach the goal given the expected return.
     - Show a recommended plan and a chart of expected portfolio value over time.
5. Interpret results
   - "Required contribution" is the periodic amount to reach the inflation-adjusted target.
   - If the current savings alone are sufficient (given growth), it will reflect a $0 required contribution.
   - The chart includes the contribution schedule and investment growth.

### Formulas Used (Simplified)
- Future value of lump sum: `FV = PV * (1 + r_per)^(n)`
- Future value of annuity (contributions): `FV_ann = PMT * [((1 + r_per)^n - 1) / r_per]`
- Solve for payment (PMT) to reach target: `PMT = (FV_target - FV_lump) * r_per / ((1 + r_per)^n - 1)`
- Inflation adjustment: `FV_target = TargetToday * (1 + inflation)^years`
- `r_per` is periodic rate (annual_return / periods_per_year), `n` is total periods.

### Disclaimer
This tool is for educational purposes only and is not financial advice. Use your own discretion and judgement. Investment returns are not guaranteed. Adjust assumptions to match your risk tolerance and local costs.

### Development
Project is plain HTML/CSS/JS, no build step.

Structure:
```
index.html
styles.css
app.js
assets/
```

Run a local server (optional):
```
python3 -m http.server 8080
```

### Accessibility and Browser Support
- Works in modern evergreen browsers.
- Keyboard accessible form and controls.

### License
MIT

# Investment-Reccommendor