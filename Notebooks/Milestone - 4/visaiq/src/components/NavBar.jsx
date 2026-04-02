import { Brain, Globe, BarChart2, TrendingUp } from 'lucide-react';
import { S } from '../theme.js';

const TABS = [
  { id: 'predict',   icon: <Globe size={15} />,      label: 'Predictor' },
  { id: 'analytics', icon: <BarChart2 size={15} />,  label: 'Analytics' },
  { id: 'insights',  icon: <TrendingUp size={15} />, label: 'Insights'  },
];

export default function NavBar({ active, onChange }) {
  return (
    <header style={{
      background: S.bg1,
      borderBottom: `1px solid ${S.border}`,
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      <div style={{
        maxWidth: '1200px', margin: '0 auto', padding: '0 1.5rem',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        height: '60px',
      }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '36px', height: '36px', borderRadius: '9px',
            background: `linear-gradient(135deg, ${S.gold}, ${S.cyan})`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Brain size={18} color={S.bg0} />
          </div>
          <div>
            <span style={{ fontSize: '15px', fontWeight: '700', color: S.goldL, letterSpacing: '2px' }}>
              VISAIQ
            </span>
            <span style={{ fontSize: '9px', color: S.txt3, letterSpacing: '2px', display: 'block', lineHeight: 1 }}>
              AI PREDICTION ENGINE
            </span>
          </div>
        </div>

        {/* Tabs */}
        <nav style={{ display: 'flex', gap: '4px' }}>
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => onChange(t.id)}
              style={{
                background:   active === t.id ? S.bg3 : 'transparent',
                border:       active === t.id ? `1px solid ${S.borderHi}` : '1px solid transparent',
                borderRadius: '8px',
                padding:      '8px 14px',
                cursor:       'pointer',
                color:        active === t.id ? S.goldL : S.txt2,
                fontFamily:   'inherit',
                fontSize:     '11px',
                letterSpacing:'1px',
                display:      'flex', alignItems: 'center', gap: '6px',
                transition:   'all 0.2s',
              }}
            >
              {t.icon}{t.label}
            </button>
          ))}
        </nav>

        {/* Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '10px', color: S.green }}>
          <div style={{
            width: '6px', height: '6px', borderRadius: '50%',
            background: S.green,
            animation: 'pulse-dot 2s ease-in-out infinite',
          }} />
          MODEL LIVE
        </div>
      </div>
    </header>
  );
}
