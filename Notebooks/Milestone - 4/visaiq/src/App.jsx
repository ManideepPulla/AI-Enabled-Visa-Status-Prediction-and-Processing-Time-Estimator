import { useState, useCallback } from 'react';
import { S } from './theme.js';
import TrainingScreen from './components/TrainingScreen.jsx';
import NavBar         from './components/NavBar.jsx';
import PredictTab     from './components/PredictTab.jsx';
import AnalyticsTab   from './components/AnalyticsTab.jsx';
import InsightsTab    from './components/InsightsTab.jsx';

export default function App() {
  const [trained, setTrained] = useState(false);
  const [tab,     setTab]     = useState('predict');
  const done = useCallback(() => setTrained(true), []);

  if (!trained) return <TrainingScreen onComplete={done} />;

  return (
    <div style={{ minHeight: '100vh', background: S.bg0, color: S.txt }}>
      <NavBar active={tab} onChange={setTab} />

      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 1.5rem 4rem' }}>
        {tab === 'predict'   && <PredictTab />}
        {tab === 'analytics' && <AnalyticsTab />}
        {tab === 'insights'  && <InsightsTab />}
      </main>

      <footer style={{ borderTop: `1px solid ${S.border}`, background: S.bg1, padding: '1rem 1.5rem' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '10px', color: S.txt3 }}>
          <span>VISAIQ Neural Engine v2.1 · Trained on 10,000 visa records · For informational purposes only</span>
          <span style={{ color: S.green }}>● All systems operational</span>
        </div>
      </footer>
    </div>
  );
}
