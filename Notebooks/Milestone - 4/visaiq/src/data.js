// ─── Countries ────────────────────────────────────────────────────────────────

export const COUNTRIES = [
  { id: 0,  name: "India",          flag: "🇮🇳", region: "South Asia",    base: 38, var: 0.28 },
  { id: 1,  name: "China",          flag: "🇨🇳", region: "East Asia",     base: 42, var: 0.30 },
  { id: 2,  name: "Nigeria",        flag: "🇳🇬", region: "West Africa",   base: 58, var: 0.38 },
  { id: 3,  name: "Brazil",         flag: "🇧🇷", region: "South America", base: 28, var: 0.24 },
  { id: 4,  name: "Mexico",         flag: "🇲🇽", region: "North America", base: 21, var: 0.20 },
  { id: 5,  name: "Philippines",    flag: "🇵🇭", region: "SE Asia",       base: 32, var: 0.26 },
  { id: 6,  name: "United Kingdom", flag: "🇬🇧", region: "Europe",        base: 13, var: 0.16 },
  { id: 7,  name: "Germany",        flag: "🇩🇪", region: "Europe",        base: 11, var: 0.14 },
  { id: 8,  name: "Pakistan",       flag: "🇵🇰", region: "South Asia",    base: 65, var: 0.42 },
  { id: 9,  name: "Bangladesh",     flag: "🇧🇩", region: "South Asia",    base: 52, var: 0.36 },
  { id: 10, name: "Ethiopia",       flag: "🇪🇹", region: "East Africa",   base: 62, var: 0.40 },
  { id: 11, name: "Vietnam",        flag: "🇻🇳", region: "SE Asia",       base: 29, var: 0.25 },
  { id: 12, name: "South Korea",    flag: "🇰🇷", region: "East Asia",     base: 15, var: 0.17 },
  { id: 13, name: "Colombia",       flag: "🇨🇴", region: "South America", base: 35, var: 0.28 },
  { id: 14, name: "Egypt",          flag: "🇪🇬", region: "North Africa",  base: 48, var: 0.35 },
];

// ─── Visa Types ───────────────────────────────────────────────────────────────

export const VISA_TYPES = [
  { id: 0, name: "Tourist",  icon: "🏖️",  mult: 0.78, approvalBase: 87, desc: "Short-term leisure visit" },
  { id: 1, name: "Student",  icon: "🎓",  mult: 1.22, approvalBase: 79, desc: "Educational enrollment" },
  { id: 2, name: "Work",     icon: "💼",  mult: 1.55, approvalBase: 68, desc: "Employment authorization" },
  { id: 3, name: "Business", icon: "🤝",  mult: 1.08, approvalBase: 82, desc: "Commercial activities" },
  { id: 4, name: "Family",   icon: "👨‍👩‍👧", mult: 1.32, approvalBase: 74, desc: "Family reunification" },
];

// ─── Processing Offices ───────────────────────────────────────────────────────

export const OFFICES = [
  { id: 0, name: "Washington D.C.", eff: 1.00, backlog: "Low" },
  { id: 1, name: "New York",        eff: 1.22, backlog: "High" },
  { id: 2, name: "Los Angeles",     eff: 1.12, backlog: "Medium" },
  { id: 3, name: "Chicago",         eff: 0.95, backlog: "Low" },
  { id: 4, name: "Houston",         eff: 0.88, backlog: "Very Low" },
  { id: 5, name: "San Francisco",   eff: 1.06, backlog: "Medium" },
];

// ─── Seasonal Multipliers (Jan–Dec) ──────────────────────────────────────────

export const SEASONAL = [1.28, 1.15, 1.05, 0.92, 0.88, 1.02, 1.12, 1.42, 1.26, 1.00, 0.90, 1.50];

export const MONTHS = [
  "January", "February", "March",     "April",   "May",      "June",
  "July",    "August",   "September", "October", "November", "December",
];

// ─── Pre-computed Analytics Data ─────────────────────────────────────────────

export const TYPE_DATA = [
  { type: "Tourist",  days: 21, approvals: 87, volume: 4231, color: "#22d3ee" },
  { type: "Student",  days: 38, approvals: 79, volume: 2847, color: "#a855f7" },
  { type: "Work",     days: 52, approvals: 68, volume: 1923, color: "#c8952a" },
  { type: "Business", days: 28, approvals: 83, volume: 1654, color: "#22c55e" },
  { type: "Family",   days: 45, approvals: 74, volume: 1289, color: "#f59e0b" },
];

export const SEASONAL_DATA = MONTHS.map((m, i) => ({
  m: m.slice(0, 3),
  days: Math.round(25 * SEASONAL[i]),
  apps: [842, 756, 868, 923, 1024, 1156, 1289, 1423, 1187, 989, 845, 987][i],
  approvalRate: [76, 78, 82, 86, 88, 85, 82, 75, 77, 83, 85, 72][i],
}));

export const COUNTRY_CHART = COUNTRIES
  .map(c => ({
    name: c.name.length > 11 ? c.name.split(" ").slice(-1)[0] : c.name,
    days: c.base,
    approval: Math.round(88 - c.var * 62),
    flag: c.flag,
  }))
  .sort((a, b) => a.days - b.days);
