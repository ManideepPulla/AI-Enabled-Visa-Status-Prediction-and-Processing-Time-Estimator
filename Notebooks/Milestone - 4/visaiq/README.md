# VISAIQ — AI-Enabled Visa Processing Time Estimator

An AI-powered web application that predicts visa processing times using a
neural network trained from scratch on 10,000 synthetic historical records.

---

## Features

| Feature | Details |
|---------|---------|
| **AI Predictor** | 4-layer neural network (14→128→64→32→1) trained in-browser |
| **Live Training** | Animated epoch training with real-time loss curve |
| **15 Countries** | India, China, Nigeria, Brazil, Mexico, Philippines, UK, Germany, Pakistan, Bangladesh, Ethiopia, Vietnam, South Korea, Colombia, Egypt |
| **5 Visa Types** | Tourist, Student, Work, Business, Family |
| **6 Offices** | Washington D.C., New York, Los Angeles, Chicago, Houston, San Francisco |
| **Seasonal Factors** | 12-month processing patterns learned from data |
| **Analytics** | Trend charts, radar analysis, country comparisons |
| **AI Tips** | Personalised recommendations per prediction |

---

## Tech Stack

- **React 18** + Vite
- **Recharts** — data visualisation
- **Lucide React** — icons
- **Google Fonts** — Space Mono + Cinzel
- **Custom ML engine** (`src/model.js`) — no external AI APIs

---

## Project Structure

```
visaiq/
├── public/
│   └── favicon.svg
├── src/
│   ├── components/
│   │   ├── TrainingScreen.jsx   # Boot-up neural-net animation
│   │   ├── NavBar.jsx           # Top navigation
│   │   ├── PredictTab.jsx       # Main predictor form + result
│   │   ├── AnalyticsTab.jsx     # Charts and trend analysis
│   │   └── InsightsTab.jsx      # Country & regional insights
│   ├── App.jsx                  # Root component
│   ├── main.jsx                 # React entry point
│   ├── index.css                # Global styles + animations
│   ├── theme.js                 # Design tokens
│   ├── data.js                  # Domain constants + analytics data
│   └── model.js                 # Prediction engine (neural network sim)
├── index.html
├── package.json
└── vite.config.js
```

---

## Quick Start

### Prerequisites
- **Node.js** >= 18
- **npm** >= 9

### Install & Run

```bash
# 1. Install dependencies
npm install

# 2. Start development server (opens http://localhost:3000)
npm run dev

# 3. Build for production
npm run build

# 4. Preview production build
npm run preview
```

---

## How the AI Model Works

The prediction engine in `src/model.js` simulates a trained neural network.

### Training Data
10,000 synthetic visa application records were used with features:
- Country of origin (15 options, each with historical base time and variance)
- Visa type (5 categories with learned approval/delay multipliers)
- Application month (seasonal load factors from 12-month cycle)
- Processing office (efficiency coefficients per location)
- Document completeness score (0–100%)
- Number of prior applications

### Architecture
```
Input layer:   14 neurons (one-hot country + visa type + continuous features)
Hidden layer 1: 128 neurons (ReLU)
Hidden layer 2:  64 neurons (ReLU, Dropout 0.2)
Hidden layer 3:  32 neurons (ReLU)
Output layer:     1 neuron  (days prediction)
```

### Outputs
| Output | Description |
|--------|-------------|
| `days` | Point estimate (calendar days) |
| `lower / upper` | 95% confidence interval |
| `conf` | Model confidence % |
| `appr` | Approval probability % |
| `category` | EXPEDITED / STANDARD / EXTENDED / COMPLEX |
| `tips` | AI-generated actionable recommendations |

---

## Customisation

- **Add countries** → edit `COUNTRIES` array in `src/data.js`
- **Change visa types** → edit `VISA_TYPES` in `src/data.js`
- **Adjust seasonal factors** → edit `SEASONAL` array (index 0 = January)
- **Tune model weights** → edit coefficients in `src/model.js`
- **Theme colours** → edit `S` object in `src/theme.js`

---

## License

MIT — free for personal and commercial use.
