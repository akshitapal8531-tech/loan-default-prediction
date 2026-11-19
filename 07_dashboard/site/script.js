// Small page interactions are kept dependency-free so the dashboard works offline.
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const revealObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((item) => revealObserver.observe(item));

function animateCounter(element) {
  const target = Number(element.dataset.target);
  if (!Number.isFinite(target)) return;
  if (reduceMotion) {
    element.textContent = String(target);
    return;
  }
  const start = performance.now();
  const duration = 1100;
  const tick = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    element.textContent = String(Math.round(target * (1 - (1 - progress) ** 3)));
    if (progress < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
const counterObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      animateCounter(entry.target);
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.8 });
document.querySelectorAll(".counter").forEach((counter) => counterObserver.observe(counter));

const menuToggle = document.querySelector(".menu-toggle");
const navLinks = document.querySelector(".nav-links");
menuToggle.addEventListener("click", () => {
  const isOpen = menuToggle.getAttribute("aria-expanded") !== "true";
  menuToggle.setAttribute("aria-expanded", String(isOpen));
  navLinks.classList.toggle("is-open", isOpen);
});
navLinks.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
  menuToggle.setAttribute("aria-expanded", "false");
  navLinks.classList.remove("is-open");
}));

const presets = {
  safe: {
    Age: 42, Income: 110000, LoanAmount: 16000, CreditScore: 790,
    MonthsEmployed: 108, NumCreditLines: 5, InterestRate: 6.5,
    LoanTerm: 36, DTIRatio: 0.18, Education: "Master's",
    EmploymentType: "Full-time", MaritalStatus: "Married", HasMortgage: "Yes",
    HasDependents: "No", LoanPurpose: "Home", HasCoSigner: "No"
  },
  risky: {
    Age: 24, Income: 28000, LoanAmount: 42000, CreditScore: 540,
    MonthsEmployed: 5, NumCreditLines: 8, InterestRate: 19,
    LoanTerm: 60, DTIRatio: 0.72, Education: "High School",
    EmploymentType: "Part-time", MaritalStatus: "Single", HasMortgage: "No",
    HasDependents: "Yes", LoanPurpose: "Business", HasCoSigner: "No"
  }
};

document.querySelectorAll(".preset-button").forEach((button) => {
  button.addEventListener("click", () => {
    const values = presets[button.dataset.preset];
    Object.entries(values).forEach(([name, value]) => {
      document.querySelector(`[name="${name}"]`).value = value;
    });
  });
});

const form = document.querySelector("#risk-form");
const submitButton = form.querySelector(".submit-button");
const emptyResult = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");
const apiError = document.querySelector("#api-error");
const gauge = document.querySelector("#probability-gauge");

function showFriendlyError() {
  emptyResult.hidden = true;
  resultContent.hidden = true;
  apiError.hidden = false;
}

function riskAdvice(level) {
  if (level === "Low") return "This profile shows a lower estimated risk. Continue to review the full application and verify its details.";
  if (level === "Medium") return "This profile has some signals worth a closer look. Consider reviewing affordability and supporting details.";
  return "This profile shows a higher estimated risk. Review the application carefully and consider whether additional support or verification is appropriate.";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;

  const payload = {};
  new FormData(form).forEach((value, key) => {
    const field = form.elements.namedItem(key);
    payload[key] = field instanceof HTMLInputElement && field.type === "number" ? Number(value) : value;
  });

  submitButton.disabled = true;
  submitButton.classList.add("is-loading");
  submitButton.querySelector(".submit-label").textContent = "Estimating…";
  apiError.hidden = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok || typeof result.default_probability !== "number") {
      throw new Error(result.detail || result.error || "Prediction failed");
    }
    const probability = Math.max(0, Math.min(1, result.default_probability));
    const percent = Math.round(probability * 100);
    const reportedRisk = String(result.risk_level || "").toLowerCase();
    const level = ["low", "medium", "high"].includes(reportedRisk)
      ? reportedRisk[0].toUpperCase() + reportedRisk.slice(1)
      : probability >= 0.7 ? "High" : probability >= 0.4 ? "Medium" : "Low";

    emptyResult.hidden = true;
    apiError.hidden = true;
    resultContent.hidden = false;
    document.querySelector("#probability-value").textContent = `${percent}%`;
    gauge.setAttribute("aria-label", `Default probability ${percent} percent`);
    gauge.style.setProperty("--progress", `${probability * 360}deg`);
    const badge = document.querySelector("#risk-badge");
    badge.textContent = level;
    badge.className = `risk-badge risk-${level.toLowerCase()}`;
    document.querySelector("#risk-advice").textContent = riskAdvice(level);
    document.querySelector("#prediction-flag").textContent = result.default_prediction
      ? "The model flagged a possible default."
      : "The model did not flag a likely default.";
  } catch (error) {
    showFriendlyError();
  } finally {
    submitButton.disabled = false;
    submitButton.classList.remove("is-loading");
    submitButton.querySelector(".submit-label").textContent = "Estimate risk";
  }
});

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",").map((value) => value.trim());
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, (values[index] || "").trim()]));
  });
}

function statusClass(status) {
  const normalized = status.toLowerCase();
  if (normalized.includes("significant") || normalized.includes("high")) return "high";
  if (normalized.includes("moderate")) return "moderate";
  return "";
}

function renderDrift(rows, tableId, chartId) {
  const table = document.querySelector(`#${tableId}`);
  const chart = document.querySelector(`#${chartId}`);
  if (!rows.length) {
    table.innerHTML = '<tr><td colspan="3" class="table-loading">No drift data available.</td></tr>';
    chart.innerHTML = '<span class="table-loading">No chart data available.</span>';
    return;
  }
  const maxPsi = Math.max(...rows.map((row) => Number(row.psi) || 0), 0.05);
  table.innerHTML = rows.map((row) => {
    const state = statusClass(row.status || "");
    const label = row.status || "stable";
    return `<tr><td>${escapeHtml(row.feature)}</td><td>${Number(row.psi).toFixed(4)}</td><td><span class="table-pill ${state}">${escapeHtml(label)}</span></td></tr>`;
  }).join("");
  chart.innerHTML = rows.map((row) => {
    const psi = Number(row.psi) || 0;
    const height = Math.max(2, Math.min(100, psi / maxPsi * 100));
    const state = statusClass(row.status || "");
    return `<div class="bar-item" title="${escapeHtml(row.feature)}: ${psi.toFixed(4)} PSI"><span class="bar ${state === "high" ? "bar-high" : state === "moderate" ? "bar-moderate" : ""}" style="height:${height}%"></span><span class="bar-label">${escapeHtml(row.feature)}</span></div>`;
  }).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

async function loadDriftReport(filename, tableId, chartId) {
  try {
    const response = await fetch(`/reports/${filename}`);
    if (!response.ok) throw new Error(`Could not load ${filename}`);
    renderDrift(parseCsv(await response.text()), tableId, chartId);
  } catch (error) {
    document.querySelector(`#${tableId}`).innerHTML = '<tr><td colspan="3" class="table-loading">Report unavailable.</td></tr>';
    document.querySelector(`#${chartId}`).innerHTML = '<span class="table-loading">Chart unavailable.</span>';
  }
}

loadDriftReport("drift_normal.csv", "normal-table", "normal-chart");
loadDriftReport("drift_simulated.csv", "simulated-table", "simulated-chart");
