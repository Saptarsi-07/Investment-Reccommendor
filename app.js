"use strict";

const plannerForm = document.getElementById("planner-form");
const resultsEl = document.getElementById("results");
const emptyStateEl = document.getElementById("emptyState");
const fvTargetEl = document.getElementById("fvTarget");
const requiredContributionEl = document.getElementById("requiredContribution");
const totalContributionsEl = document.getElementById("totalContributions");
const growthEl = document.getElementById("growth");
const recommendationsEl = document.getElementById("recommendations");

let chartInstance = null;

plannerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const formData = new FormData(plannerForm);

    const goalName = String(formData.get("goalName") || "Your Goal");
    const targetAmount = Math.max(0, Number(formData.get("targetAmount")) || 0);
    const years = Math.max(0, Number(formData.get("years")) || 0);
    const currentSavings = Math.max(0, Number(formData.get("currentSavings")) || 0);
    const returnRateAnnualPct = Math.max(0, Number(formData.get("returnRate")) || 0);
    const inflationAnnualPct = Math.max(0, Number(formData.get("inflationRate")) || 0);
    const frequency = String(formData.get("frequency")) === "weekly" ? "weekly" : "monthly";

    const periodsPerYear = frequency === "weekly" ? 52 : 12;
    const totalPeriods = Math.round(years * periodsPerYear);
    if (totalPeriods <= 0 || targetAmount <= 0) {
        showEmpty("Please enter a positive target and time horizon.");
        return;
    }

    const rAnnual = returnRateAnnualPct / 100;
    const iAnnual = inflationAnnualPct / 100;
    const rPer = rAnnual / periodsPerYear;

    // Inflation adjust target to future value at goal date
    const fvTarget = targetAmount * Math.pow(1 + iAnnual, years);
    // Future value of current savings
    const fvCurrent = currentSavings * Math.pow(1 + rPer, totalPeriods);

    // If no growth per period (rPer==0), use linear contributions
    let paymentPerPeriod = 0;
    if (rPer === 0) {
        const shortfall = Math.max(0, fvTarget - fvCurrent);
        paymentPerPeriod = shortfall / totalPeriods;
    } else {
        const numerator = Math.max(0, fvTarget - fvCurrent) * rPer;
        const denominator = Math.pow(1 + rPer, totalPeriods) - 1;
        paymentPerPeriod = denominator > 0 ? numerator / denominator : 0;
    }

    paymentPerPeriod = Math.max(0, paymentPerPeriod);

    // Build schedule and compute totals
    const labels = [];
    const portfolioValues = [];
    const contributionCumulative = [];
    let value = currentSavings;
    let totalContrib = 0;
    for (let p = 1; p <= totalPeriods; p++) {
        // contribution at period end
        totalContrib += paymentPerPeriod;
        value = value * (1 + rPer) + paymentPerPeriod;
        if (p % periodsPerYear === 0 || p === totalPeriods) {
            const yr = Math.ceil(p / periodsPerYear);
            labels.push(`Year ${yr}`);
            portfolioValues.push(value);
            contributionCumulative.push(totalContrib);
        }
    }

    const expectedGrowth = Math.max(0, portfolioValues[portfolioValues.length - 1] - totalContrib - currentSavings);

    // Update UI numbers
    fvTargetEl.textContent = formatCurrency(fvTarget);
    requiredContributionEl.textContent = `${formatCurrency(paymentPerPeriod)} per ${frequency}`;
    totalContributionsEl.textContent = formatCurrency(totalContrib);
    growthEl.textContent = formatCurrency(expectedGrowth);

    // Strategy recommendations
    const recs = buildRecommendations({
        goalName,
        paymentPerPeriod,
        frequency,
        years,
        periodsPerYear,
        fvTarget,
        currentSavings,
        returnRateAnnualPct,
        inflationAnnualPct
    });
    recommendationsEl.innerHTML = recs.map(r => `<p>• ${r}</p>`).join("");

    // Show results and chart
    resultsEl.classList.remove("hidden");
    emptyStateEl.classList.add("hidden");
    renderChart(labels, portfolioValues, contributionCumulative, fvTarget);
});

function showEmpty(message) {
    emptyStateEl.textContent = message;
    emptyStateEl.classList.remove("hidden");
    resultsEl.classList.add("hidden");
}

function formatCurrency(amount) {
    return new Intl.NumberFormat(undefined, { style: "currency", currency: guessCurrency(), maximumFractionDigits: 0 }).format(amount);
}

function guessCurrency() {
    // Best-effort: try browser locale currency; fallback USD
    try {
        const parts = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" }).resolvedOptions();
        return parts.currency || "USD";
    } catch {
        return "USD";
    }
}

function buildRecommendations(ctx) {
    const recs = [];
    const { paymentPerPeriod, frequency, years, periodsPerYear, fvTarget, currentSavings, returnRateAnnualPct, inflationAnnualPct } = ctx;

    if (paymentPerPeriod === 0) {
        recs.push("You're already on track. Consider reducing risk as the goal nears.");
    } else {
        recs.push(`Set up auto-invest: ${formatCurrency(paymentPerPeriod)} per ${frequency}.`);
    }

    if (years <= 2) {
        recs.push("Short horizon: prefer lower volatility options (high-yield savings, short-term bonds).");
    } else if (years <= 5) {
        recs.push("Medium horizon: balanced mix (bonds + broad equity index funds).");
    } else {
        recs.push("Long horizon: tilt toward diversified equities; de-risk 1-2 years before goal.");
    }

    if (inflationAnnualPct >= 4) {
        recs.push("High inflation assumption: review target costs more frequently.");
    }
    if (returnRateAnnualPct >= 10) {
        recs.push("Optimistic return assumption: stress-test with lower returns (e.g., 5-7%).");
    }
    if ((paymentPerPeriod * periodsPerYear) > (0.3 * (fvTarget / years))) {
        recs.push("Contributions are high relative to goal: extend timeline or lower target if needed.");
    }
    if (currentSavings === 0) {
        recs.push("Start with a small emergency buffer before investing toward the goal.");
    }
    return recs;
}

function renderChart(labels, portfolioValues, contributionCumulative, fvTarget) {
    const ctx = document.getElementById("growthChart");
    if (!ctx) return;
    if (chartInstance) {
        chartInstance.destroy();
    }
    chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels,
            datasets: [
                {
                    label: "Projected Portfolio",
                    data: portfolioValues,
                    borderColor: "#6ea8fe",
                    backgroundColor: "rgba(110,168,254,0.15)",
                    tension: 0.2,
                },
                {
                    label: "Total Contributions",
                    data: contributionCumulative,
                    borderColor: "#22c55e",
                    backgroundColor: "rgba(34,197,94,0.15)",
                    tension: 0.2,
                },
                {
                    label: "Target (inflation-adjusted)",
                    data: labels.map(() => fvTarget),
                    borderColor: "#ef4444",
                    borderDash: [6, 6],
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#e6edf6" } },
                tooltip: { mode: "index", intersect: false }
            },
            scales: {
                x: {
                    ticks: { color: "#9aa4b2" },
                    grid: { color: "#233044" }
                },
                y: {
                    ticks: { color: "#9aa4b2" },
                    grid: { color: "#233044" }
                }
            }
        }
    });
}

