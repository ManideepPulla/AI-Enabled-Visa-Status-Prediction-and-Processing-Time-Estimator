import { useState, useEffect, useCallback } from 'react';
import { Brain } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import { S, card } from '../theme.js';

const TOTAL_EPOCHS = 50;

export default function TrainingScreen({ onComplete }) {
  const [epoch,     setEpoch]     = useState(0);
  const [loss,      setLoss]      = useState(2.847);
  const [acc,       setAcc]       = useState(0.312);
  const [lossHist,  setLossHist]  = useState([{ e: 0, l: 2.847 }]);
  const [samples,   setSamples]   = useState(0);
  const [phase,     setPhase]     = useState('loading'); // loading | training | done

  // Phase 1 — data ingestion animation
  useEffect(() => {
    const t = setInterval(() => {
      setSamples(prev => {
        const next = prev + Math.floor(Math.random() * 380 + 120);
        if (next >= 10000) {
          clearInterval(t);
          setPhase('training');
          return 10000;
        }
        return next;
      });
    }, 45);
    return () => clearInterval(t);
  }, []);

  // Phase 2 — epoch training animation
  useEffect(() => {
    if (phase !== 'training') return;
    const t = setInterval(() => {
      setEpoch(e => {
        if (e >= TOTAL_EPOCHS) {
          clearInterval(t);
          setPhase('done');
          setTimeout(onComplete, 700);
          return TOTAL_EPOCHS;
        }
        const ne = e + 1;
        const nl = Math.max(0.065, 2.847 * Math.exp(-ne * 0.085) + 0.035 * Math.random());
        const na = Math.min(0.952, 0.312 + 0.640 * (1 - Math.exp(-ne * 0.078)));
        setLoss(+nl.toFixed(4));
        setAcc(+na.toFixed(4));
        setLossHist(h => [...h, { e: ne, l: +nl.toFixed(4) }]);
        return ne;
      });
    }, 65);
    return () => clearInterval(t);
  }, [phase, onComplete]);

  const pct = phase === 'loading'
    ? (samples / 10000) * 45
    : 45 + (epoch / TOTAL_EPOCHS) * 55;

  return (
    <div style={{
      minHeight: '100vh', background: S.bg0,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '2rem',
    }}>
      <div style={{ maxWidth: '580px', width: '100%' }}>

        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '2.5rem' }}>
          <div style={{
            width: '52px', height: '52px', borderRadius: '14px', flexShrink: 0,
            background: `linear-gradient(135deg, ${S.gold}, ${S.cyan})`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <Brain size={28} color={S.bg0} />
          </div>
          <div>
            <div style={{ fontSize: '20px', fontWeight: '700', color: S.goldL, letterSpacing: '3px' }}>VISAIQ</div>
            <div style={{ fontSize: '10px', color: S.txt2, letterSpacing: '4px', marginTop: '2px' }}>
              PREDICTIVE ANALYTICS ENGINE
            </div>
          </div>
        </div>

        {/* Status text */}
        <h2 style={{ color: S.txt, fontSize: '18px', fontWeight: 'normal', marginBottom: '6px' }}>
          {phase === 'loading'   ? 'Ingesting Historical Data'
           : phase === 'training' ? 'Training Neural Network'
           : '✓ Model Ready'}
        </h2>
        <p style={{ color: S.txt2, fontSize: '11px', letterSpacing: '1px', marginBottom: '1.5rem' }}>
          {phase === 'loading'
            ? `${Math.min(samples, 10000).toLocaleString()} / 10,000 visa records processed`
            : phase === 'training'
            ? `Epoch ${epoch}/${TOTAL_EPOCHS} · Adam optimizer · MSE loss convergence`
            : 'Neural network calibrated — ready for predictions'}
        </p>

        {/* Progress bar */}
        <div style={{ background: S.bg2, borderRadius: '4px', height: '3px', marginBottom: '2rem', overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            background: `linear-gradient(90deg, ${S.gold}, ${S.cyan})`,
            width: `${pct}%`,
            transition: 'width 0.08s ease',
            borderRadius: '4px',
          }} />
        </div>

        {/* Live metrics */}
        {phase === 'training' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', marginBottom: '1.5rem' }}>
            {[
              { label: 'TRAIN LOSS', val: loss.toFixed(4),         color: S.red   },
              { label: 'ACCURACY',   val: `${(acc*100).toFixed(1)}%`, color: S.green },
              { label: 'EPOCH',      val: `${epoch}/${TOTAL_EPOCHS}`, color: S.cyan  },
            ].map(m => (
              <div key={m.label} style={{ ...card, padding: '14px 10px', textAlign: 'center' }}>
                <div style={{ fontSize: '8px', color: S.txt3, letterSpacing: '2px', marginBottom: '6px' }}>
                  {m.label}
                </div>
                <div style={{ fontSize: '20px', color: m.color, fontWeight: '700' }}>{m.val}</div>
              </div>
            ))}
          </div>
        )}

        {/* Loss curve */}
        {lossHist.length > 3 && (
          <div style={{ ...card, padding: '14px' }}>
            <div style={{ fontSize: '9px', color: S.txt3, letterSpacing: '2px', marginBottom: '10px' }}>
              TRAINING LOSS CURVE
            </div>
            <ResponsiveContainer width="100%" height={80}>
              <LineChart data={lossHist} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                <Line type="monotone" dataKey="l" stroke={S.gold} dot={false} strokeWidth={2} />
                <XAxis dataKey="e" hide />
                <YAxis hide domain={[0, 3]} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Architecture info */}
        <div style={{
          marginTop: '1.5rem', fontSize: '10px', color: S.txt3,
          letterSpacing: '1px', lineHeight: '1.9', textAlign: 'center',
        }}>
          Architecture: 14 → 128 → 64 → 32 → 1 neurons · ReLU activations · Dropout 0.2
          <br />
          Dataset: 10,000 records · 15 countries · 5 visa types · 12 seasonal factors
        </div>
      </div>
    </div>
  );
}
