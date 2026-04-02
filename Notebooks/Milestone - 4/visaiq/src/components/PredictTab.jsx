import { useState } from 'react';
import {
  Brain, Shield, FileText, User, MapPin, Calendar,
  Star, Zap, Info, RefreshCw, CheckCircle,
} from 'lucide-react';
import { S, card, inputStyle, btnOutline } from '../theme.js';
import { COUNTRIES, VISA_TYPES, OFFICES, SEASONAL, MONTHS } from '../data.js';
import { predict } from '../model.js';

// ─── Reusable label ───────────────────────────────────────────────────────────
function FieldLabel({ icon, children, right }) {
  return (
    <div style={{
      fontSize: '10px', color: S.txt2, letterSpacing: '2px',
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      marginBottom: '6px',
    }}>
      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        {icon}{children}
      </span>
      {right}
    </div>
  );
}

// ─── Result panel ─────────────────────────────────────────────────────────────
function ResultPanel({ result, onReset }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', animation: 'fadeIn 0.4s ease' }}>

      {/* Main result */}
      <div style={{ ...card, border: `1px solid ${result.catColor}44`, position: 'relative', overflow: 'hidden' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: `linear-gradient(90deg, ${result.catColor}, transparent)` }} />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
          <div>
            <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '3px', marginBottom: '6px' }}>ESTIMATED PROCESSING TIME</div>
            <div style={{ fontSize: '54px', fontWeight: '700', color: result.catColor, lineHeight: 1 }}>
              {result.days}
            </div>
            <div style={{ fontSize: '14px', color: S.txt2, marginTop: '4px' }}>CALENDAR DAYS</div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{
              background: `${result.catColor}22`, border: `1px solid ${result.catColor}55`,
              borderRadius: '6px', padding: '6px 12px', color: result.catColor,
              fontSize: '11px', letterSpacing: '2px', marginBottom: '8px',
            }}>
              {result.category}
            </div>
            <div style={{ fontSize: '11px', color: S.txt2 }}>
              Range: <span style={{ color: S.txt }}>{result.lower}–{result.upper} days</span>
            </div>
          </div>
        </div>

        {/* Confidence bar */}
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: S.txt3, marginBottom: '6px' }}>
            <span>MODEL CONFIDENCE</span>
            <span style={{ color: S.txt }}>{result.conf}%</span>
          </div>
          <div style={{ background: S.bg3, borderRadius: '3px', height: '6px' }}>
            <div style={{ height: '100%', width: `${result.conf}%`, background: `linear-gradient(90deg, ${S.gold}, ${S.cyan})`, borderRadius: '3px', transition: 'width 0.6s ease' }} />
          </div>
        </div>

        {/* Mini stats */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
          {[
            { label: 'APPROVAL PROB', val: `${result.appr}%`, color: result.appr >= 75 ? S.green : result.appr >= 60 ? S.amber : S.red },
            { label: 'SIMILAR CASES', val: result.similar.total.toLocaleString(), color: S.cyan },
            { label: 'FASTEST SEEN',  val: `${result.similar.fastest}d`, color: S.green },
          ].map(s => (
            <div key={s.label} style={{ background: S.bg2, border: `1px solid ${S.border}`, borderRadius: '8px', padding: '10px 8px', textAlign: 'center' }}>
              <div style={{ fontSize: '8px', color: S.txt3, letterSpacing: '1.5px', marginBottom: '4px' }}>{s.label}</div>
              <div style={{ fontSize: '17px', color: s.color, fontWeight: '700' }}>{s.val}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Range visualization */}
      <div style={{ ...card }}>
        <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px', marginBottom: '16px' }}>PREDICTION RANGE DISTRIBUTION</div>
        <div style={{ position: 'relative', height: '52px' }}>
          <div style={{ position: 'absolute', top: '20px', left: '2%', right: '2%', height: '3px', background: S.bg3, borderRadius: '2px' }} />
          {/* CI band */}
          <div style={{
            position: 'absolute', top: '16px', height: '11px',
            left: '8%', right: '10%',
            background: `${S.cyan}22`, borderRadius: '3px', border: `1px solid ${S.cyan}44`,
          }} />
          {/* Point */}
          <div style={{
            position: 'absolute', top: '14px', width: '15px', height: '15px',
            background: result.catColor, borderRadius: '50%',
            left: '48%', transform: 'translateX(-50%)',
            boxShadow: `0 0 10px ${result.catColor}66`,
          }} />
          <div style={{ position: 'absolute', top: '36px', left: '0',   fontSize: '10px', color: S.txt3 }}>{result.lower}d (best)</div>
          <div style={{ position: 'absolute', top: '36px', left: '50%', fontSize: '10px', color: result.catColor, fontWeight: '700', transform: 'translateX(-50%)' }}>{result.days}d (predicted)</div>
          <div style={{ position: 'absolute', top: '36px', right: '0',  fontSize: '10px', color: S.txt3 }}>{result.upper}d (worst)</div>
        </div>
      </div>

      {/* Tips */}
      {result.tips.length > 0 && (
        <div style={{ ...card }}>
          <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px', marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Info size={12} color={S.gold} /> AI RECOMMENDATIONS
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {result.tips.map((tip, i) => (
              <div key={i} style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
                <div style={{
                  width: '20px', height: '20px', borderRadius: '50%', flexShrink: 0, marginTop: '1px',
                  background: `${S.gold}22`, border: `1px solid ${S.gold}44`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <span style={{ fontSize: '9px', color: S.gold }}>→</span>
                </div>
                <div style={{ fontSize: '12px', color: S.txt2, lineHeight: 1.6 }}>{tip}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <button onClick={onReset} style={{ ...btnOutline, width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
        <RefreshCw size={13} /> NEW PREDICTION
      </button>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function PredictTab() {
  const [form, setForm] = useState({
    countryId: '0', visaTypeId: '0', monthId: '0',
    officeId: '0', docScore: 0.85, priorApps: '0',
  });
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const handlePredict = () => {
    setResult(null);
    setLoading(true);
    setTimeout(() => {
      const r = predict({
        countryId:  +form.countryId,
        visaTypeId: +form.visaTypeId,
        monthId:    +form.monthId,
        officeId:   +form.officeId,
        docScore:    form.docScore,
        priorApps:  +form.priorApps,
      });
      setResult(r);
      setLoading(false);
    }, 950);
  };

  const country  = COUNTRIES[+form.countryId];
  const seasonal = SEASONAL[+form.monthId];

  return (
    <div>
      {/* Page hero */}
      <div style={{ textAlign: 'center', padding: '2rem 0 2.5rem' }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '8px',
          background: S.bg2, border: `1px solid ${S.border}`,
          borderRadius: '20px', padding: '6px 16px',
          fontSize: '10px', color: S.cyan, letterSpacing: '2px', marginBottom: '1rem',
        }}>
          <Zap size={12} /> AI-POWERED PREDICTION
        </div>
        <h1 style={{ fontFamily: "'Cinzel', serif", fontSize: '30px', fontWeight: '600', color: S.goldL, margin: '0 0 10px', letterSpacing: '1px', lineHeight: 1.2 }}>
          Visa Processing Time Estimator
        </h1>
        <p style={{ color: S.txt2, fontSize: '13px', maxWidth: '540px', margin: '0 auto', lineHeight: 1.7 }}>
          Our neural network — trained on 10,000+ historical applications — predicts your processing timeline
          with confidence intervals and actionable improvement tips.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>

        {/* ── Form ── */}
        <div style={{ ...card }}>
          <div style={{ fontSize: '11px', color: S.txt3, letterSpacing: '3px', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <FileText size={13} color={S.gold} /> APPLICATION PARAMETERS
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>

            {/* Country */}
            <div>
              <FieldLabel icon={<User size={11} />}>COUNTRY OF ORIGIN</FieldLabel>
              <div style={{ position: 'relative' }}>
                <span style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', fontSize: '18px', pointerEvents: 'none' }}>
                  {country.flag}
                </span>
                <select value={form.countryId} onChange={e => set('countryId', e.target.value)}
                  style={{ ...inputStyle, paddingLeft: '40px' }}>
                  {COUNTRIES.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              </div>
              <div style={{ fontSize: '10px', color: S.txt3, marginTop: '4px' }}>Region: {country.region}</div>
            </div>

            {/* Visa type */}
            <div>
              <FieldLabel icon={<Shield size={11} />}>VISA CATEGORY</FieldLabel>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: '6px' }}>
                {VISA_TYPES.map(v => (
                  <button key={v.id} onClick={() => set('visaTypeId', String(v.id))} style={{
                    background:   +form.visaTypeId === v.id ? S.bg3 : S.bg1,
                    border:       +form.visaTypeId === v.id ? `1px solid ${S.gold}` : `1px solid ${S.border}`,
                    borderRadius: '8px', padding: '10px 4px', cursor: 'pointer',
                    textAlign: 'center', transition: 'all 0.15s',
                    color: +form.visaTypeId === v.id ? S.goldL : S.txt2,
                    fontFamily: 'inherit',
                  }}>
                    <div style={{ fontSize: '18px', marginBottom: '3px' }}>{v.icon}</div>
                    <div style={{ fontSize: '9px', letterSpacing: '0.5px' }}>{v.name}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Month + Office */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div>
                <FieldLabel icon={<Calendar size={11} />}>APPLICATION MONTH</FieldLabel>
                <select value={form.monthId} onChange={e => set('monthId', e.target.value)} style={inputStyle}>
                  {MONTHS.map((m, i) => <option key={i} value={i}>{m}</option>)}
                </select>
                <div style={{ fontSize: '10px', marginTop: '4px', color: seasonal > 1.2 ? S.amber : seasonal < 0.95 ? S.green : S.txt3 }}>
                  {seasonal > 1.2 ? '⚠ Peak season' : seasonal < 0.95 ? '✓ Low season' : '◎ Normal season'}
                </div>
              </div>
              <div>
                <FieldLabel icon={<MapPin size={11} />}>PROCESSING OFFICE</FieldLabel>
                <select value={form.officeId} onChange={e => set('officeId', e.target.value)} style={inputStyle}>
                  {OFFICES.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}
                </select>
                <div style={{ fontSize: '10px', color: S.txt3, marginTop: '4px' }}>
                  Backlog: {OFFICES[+form.officeId].backlog}
                </div>
              </div>
            </div>

            {/* Document completeness */}
            <div>
              <FieldLabel
                icon={<FileText size={11} />}
                right={
                  <span style={{ color: form.docScore >= 0.9 ? S.green : form.docScore >= 0.7 ? S.amber : S.red }}>
                    {Math.round(form.docScore * 100)}%
                  </span>
                }
              >
                DOCUMENT COMPLETENESS
              </FieldLabel>
              <div style={{ position: 'relative', height: '28px', display: 'flex', alignItems: 'center' }}>
                <div style={{ position: 'absolute', width: '100%', height: '4px', background: S.bg3, borderRadius: '2px' }}>
                  <div style={{
                    height: '100%', width: `${form.docScore * 100}%`,
                    background: `linear-gradient(90deg, ${S.red}, ${S.amber}, ${S.green})`,
                    borderRadius: '2px', transition: 'width 0.1s',
                  }} />
                </div>
                <input
                  type="range" min="0.3" max="1" step="0.05"
                  value={form.docScore}
                  onChange={e => set('docScore', parseFloat(e.target.value))}
                  style={{ position: 'relative', width: '100%', background: 'transparent', height: '28px', cursor: 'pointer', zIndex: 1 }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: S.txt3, marginTop: '2px' }}>
                <span>Incomplete</span><span>Partial</span><span>Complete</span>
              </div>
            </div>

            {/* Prior applications */}
            <div>
              <FieldLabel icon={<Star size={11} />}>PRIOR APPLICATIONS</FieldLabel>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '6px' }}>
                {[0, 1, 2, 3].map(n => (
                  <button key={n} onClick={() => set('priorApps', String(n))} style={{
                    background:   +form.priorApps === n ? S.bg3 : S.bg1,
                    border:       +form.priorApps === n ? `1px solid ${S.cyan}` : `1px solid ${S.border}`,
                    borderRadius: '8px', padding: '10px', cursor: 'pointer',
                    color:        +form.priorApps === n ? S.cyan : S.txt2,
                    fontFamily:   'inherit', fontSize: '13px', transition: 'all 0.15s',
                  }}>
                    {n === 0 ? 'None' : n === 1 ? '1st' : n === 2 ? '2nd' : '3+'}
                  </button>
                ))}
              </div>
            </div>

            {/* Submit */}
            <button
              onClick={handlePredict}
              disabled={loading}
              style={{
                background:   loading ? S.bg2 : `linear-gradient(135deg, ${S.goldD}, ${S.gold})`,
                border:       `1px solid ${S.gold}`,
                borderRadius: '10px',
                padding:      '14px',
                color:        loading ? S.txt2 : S.bg0,
                fontFamily:   'inherit',
                fontSize:     '13px',
                letterSpacing:'2px',
                cursor:       loading ? 'not-allowed' : 'pointer',
                fontWeight:   '700',
                display:      'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
                transition:   'all 0.2s',
                marginTop:    '4px',
              }}
            >
              {loading
                ? <><RefreshCw size={15} style={{ animation: 'spin 1s linear infinite' }} /> ANALYZING APPLICATION...</>
                : <><Brain size={15} /> PREDICT PROCESSING TIME</>
              }
            </button>
          </div>
        </div>

        {/* ── Results ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {!result && !loading && (
            <div style={{ ...card, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '420px', gap: '16px', opacity: 0.55 }}>
              <Brain size={52} color={S.txt3} />
              <div style={{ fontSize: '13px', color: S.txt3, textAlign: 'center', lineHeight: 1.7 }}>
                Configure your application parameters<br />and click Predict to see results
              </div>
            </div>
          )}

          {loading && (
            <div style={{ ...card, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '420px', gap: '20px' }}>
              <div style={{ width: '60px', height: '60px', borderRadius: '50%', border: `2px solid ${S.border}`, borderTop: `2px solid ${S.gold}`, animation: 'spin 1s linear infinite' }} />
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '13px', color: S.goldL, letterSpacing: '1px' }}>Neural Network Processing</div>
                <div style={{ fontSize: '11px', color: S.txt2, marginTop: '6px' }}>Evaluating similar historical cases...</div>
              </div>
            </div>
          )}

          {result && !loading && (
            <ResultPanel result={result} onReset={() => setResult(null)} />
          )}
        </div>
      </div>
    </div>
  );
}
