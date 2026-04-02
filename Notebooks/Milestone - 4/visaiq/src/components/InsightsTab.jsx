import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Cell,
} from 'recharts';
import { S, card } from '../theme.js';
import { COUNTRIES, COUNTRY_CHART } from '../data.js';
import { CustomTooltip } from './AnalyticsTab.jsx';

export default function InsightsTab() {
  const [activeCountry, setActiveCountry] = useState(null);

  const radarData = [0, 1, 2, 4, 6, 8].map(id => {
    const c = COUNTRIES[id];
    return {
      name:            c.name.split(' ').slice(-1)[0],
      speed:           Math.round(100 - c.base * 0.8),
      predictability:  Math.round(100 - c.var * 100),
      approvalRate:    Math.round(88 - c.var * 62),
      volume:          [85, 78, 55, 70, 60, 50][id % 6],
      compliance:      Math.round(90 - c.var * 40),
    };
  });

  return (
    <div>
      <div style={{ padding: '1.5rem 0 2rem' }}>
        <h2 style={{ fontFamily: "'Cinzel', serif", fontSize: '24px', color: S.goldL, fontWeight: '600', margin: '0 0 6px' }}>
          Country &amp; Regional Insights
        </h2>
        <p style={{ color: S.txt2, fontSize: '12px' }}>
          Deep-dive analysis of processing patterns by country of origin and application type
        </p>
      </div>

      {/* Horizontal bar — country processing times */}
      <div style={{ ...card, marginBottom: '1.5rem' }}>
        <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px', marginBottom: '1.5rem' }}>
          AVERAGE PROCESSING TIME BY COUNTRY (days, sorted fastest → slowest)
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={COUNTRY_CHART} layout="vertical" margin={{ top: 0, right: 60, bottom: 0, left: 0 }}>
            <CartesianGrid stroke={S.border} strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fill: S.txt3, fontSize: 9, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} domain={[0, 75]} />
            <YAxis type="category" dataKey="name" width={80} tick={{ fill: S.txt2, fontSize: 10, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="days" name="Avg Days" radius={[0, 4, 4, 0]}>
              {COUNTRY_CHART.map((d, i) => (
                <Cell key={i} fill={d.days <= 20 ? S.green : d.days <= 35 ? S.cyan : d.days <= 50 ? S.amber : S.red} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap', marginTop: '12px' }}>
          {[
            { c: S.green, l: '≤20 days (Fast)'     },
            { c: S.cyan,  l: '21–35 days (Normal)'  },
            { c: S.amber, l: '36–50 days (Extended)'},
            { c: S.red,   l: '>50 days (Complex)'   },
          ].map(x => (
            <span key={x.l} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: S.txt2 }}>
              <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: x.c }} />{x.l}
            </span>
          ))}
        </div>
      </div>

      {/* Country cards */}
      <div style={{ marginBottom: '12px', fontSize: '10px', color: S.txt3, letterSpacing: '2px' }}>
        TOP SOURCE COUNTRIES — CLICK TO EXPAND DETAILS
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '1.5rem' }}>
        {COUNTRIES.slice(0, 9).map(c => {
          const approvalRate = Math.round(88 - c.var * 62);
          const risk      = c.var > 0.35 ? 'HIGH' : c.var > 0.25 ? 'MEDIUM' : 'LOW';
          const riskColor = c.var > 0.35 ? S.red  : c.var > 0.25 ? S.amber   : S.green;
          const isOpen    = activeCountry === c.id;
          return (
            <div
              key={c.id}
              onClick={() => setActiveCountry(isOpen ? null : c.id)}
              style={{
                ...card, padding: '14px', cursor: 'pointer', transition: 'all 0.2s',
                border:     isOpen ? `1px solid ${S.gold}` : `1px solid ${S.border}`,
                background: isOpen ? S.bg3 : S.bg2,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '20px' }}>{c.flag}</span>
                  <div>
                    <div style={{ fontSize: '12px', color: S.txt, fontWeight: '700' }}>{c.name.split(' ').slice(-1)[0]}</div>
                    <div style={{ fontSize: '9px', color: S.txt3 }}>{c.region}</div>
                  </div>
                </div>
                <div style={{ fontSize: '9px', background: `${riskColor}22`, border: `1px solid ${riskColor}44`, borderRadius: '4px', padding: '3px 7px', color: riskColor }}>
                  {risk}
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                <div>
                  <div style={{ fontSize: '8px', color: S.txt3, marginBottom: '2px' }}>AVG DAYS</div>
                  <div style={{ fontSize: '18px', color: c.base <= 20 ? S.green : c.base <= 40 ? S.cyan : S.amber, fontWeight: '700' }}>{c.base}</div>
                </div>
                <div>
                  <div style={{ fontSize: '8px', color: S.txt3, marginBottom: '2px' }}>APPROVAL</div>
                  <div style={{ fontSize: '18px', color: approvalRate >= 80 ? S.green : approvalRate >= 70 ? S.amber : S.red, fontWeight: '700' }}>{approvalRate}%</div>
                </div>
              </div>

              {isOpen && (
                <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: `1px solid ${S.border}`, fontSize: '10px', color: S.txt2, lineHeight: 1.8, animation: 'fadeIn 0.25s ease' }}>
                  <div>Variance index: <span style={{ color: S.txt }}>{(c.var * 100).toFixed(0)}%</span></div>
                  <div>Peak month impact: <span style={{ color: S.amber }}>+{Math.round(c.base * 0.45)} days</span></div>
                  <div>Low season saving: <span style={{ color: S.green }}>−{Math.round(c.base * 0.18)} days</span></div>
                  <div>Model confidence: <span style={{ color: S.cyan }}>{Math.round(92 - c.var * 28)}%</span></div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Radar — regional profile */}
      <div style={{ ...card }}>
        <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px', marginBottom: '1.5rem' }}>
          REGIONAL PROCESSING PROFILE — RADAR ANALYSIS
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'center' }}>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
              <PolarGrid stroke={S.border} />
              <PolarAngleAxis dataKey="name" tick={{ fill: S.txt2, fontSize: 10, fontFamily: "'Space Mono', monospace" }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: S.txt3, fontSize: 8 }} axisLine={false} />
              <Radar name="Speed"           dataKey="speed"           stroke={S.cyan}  fill={S.cyan}  fillOpacity={0.15} strokeWidth={2} />
              <Radar name="Predictability"  dataKey="predictability"  stroke={S.gold}  fill={S.gold}  fillOpacity={0.10} strokeWidth={1.5} />
              <Tooltip content={<CustomTooltip />} />
            </RadarChart>
          </ResponsiveContainer>

          <div>
            <div style={{ fontSize: '12px', color: S.txt, marginBottom: '1rem', lineHeight: 1.7 }}>
              Six major source countries compared across key model-derived metrics:
            </div>
            {[
              { label: 'Processing Speed',   desc: 'Inverse of average days — higher is faster', color: S.cyan },
              { label: 'Predictability',     desc: 'Consistency of outcomes — lower variance = higher score', color: S.gold },
              { label: 'Approval Rate',      desc: 'Historical approval probability for complete applications', color: S.green },
            ].map(m => (
              <div key={m.label} style={{ display: 'flex', gap: '10px', marginBottom: '14px', alignItems: 'flex-start' }}>
                <div style={{ width: '3px', height: '38px', background: m.color, borderRadius: '2px', flexShrink: 0, marginTop: '2px' }} />
                <div>
                  <div style={{ fontSize: '11px', color: m.color, fontWeight: '700' }}>{m.label}</div>
                  <div style={{ fontSize: '10px', color: S.txt3, lineHeight: 1.6, marginTop: '2px' }}>{m.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
