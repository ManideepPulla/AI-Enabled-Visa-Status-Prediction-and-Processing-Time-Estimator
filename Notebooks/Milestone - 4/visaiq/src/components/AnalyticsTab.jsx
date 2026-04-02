import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Cell,
} from 'recharts';
import { Clock, CheckCircle, Activity, TrendingUp } from 'lucide-react';
import { S, card } from '../theme.js';
import { TYPE_DATA, SEASONAL_DATA } from '../data.js';

// ─── Custom tooltip ────────────────────────────────────────────────────────────
export function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: S.bg2, border: `1px solid ${S.border}`, borderRadius: '8px', padding: '10px 14px', fontFamily: "'Space Mono', monospace", fontSize: '11px' }}>
      <div style={{ color: S.txt2, marginBottom: '4px' }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || S.txt, marginTop: '2px' }}>
          {p.name}: <strong>{p.value}</strong>
        </div>
      ))}
    </div>
  );
}

// ─── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({ icon, label, value, sub, color = S.cyan }) {
  return (
    <div style={{ ...card, display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px' }}>{label}</div>
        <div style={{ color, opacity: 0.7 }}>{icon}</div>
      </div>
      <div style={{ fontSize: '28px', fontWeight: '700', color, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: '11px', color: S.txt2 }}>{sub}</div>}
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────
export default function AnalyticsTab() {
  return (
    <div>
      <div style={{ padding: '1.5rem 0 2rem' }}>
        <h2 style={{ fontFamily: "'Cinzel', serif", fontSize: '24px', color: S.goldL, fontWeight: '600', margin: '0 0 6px' }}>
          Trend Analytics
        </h2>
        <p style={{ color: S.txt2, fontSize: '12px' }}>
          Aggregated insights from 10,000+ historical visa applications across all categories
        </p>
      </div>

      {/* Summary stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '14px', marginBottom: '1.5rem' }}>
        <StatCard icon={<Clock size={18}/>}        label="AVG PROCESSING TIME"    value="34d"   sub="Across all visa types"     color={S.cyan}   />
        <StatCard icon={<CheckCircle size={18}/>}  label="OVERALL APPROVAL RATE"  value="79%"   sub="Based on complete docs"    color={S.green}  />
        <StatCard icon={<Activity size={18}/>}     label="APPLICATIONS (YTD)"     value="12.9K" sub="+8.2% year over year"     color={S.amber}  />
        <StatCard icon={<TrendingUp size={18}/>}   label="FASTEST CASE ON RECORD" value="5d"    sub="Tourist — low season"      color={S.purple} />
      </div>

      {/* Row 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>

        {/* Avg processing days by type */}
        <div style={{ ...card }}>
          <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px', marginBottom: '1.5rem' }}>
            AVG PROCESSING DAYS BY VISA TYPE
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={TYPE_DATA} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <CartesianGrid stroke={S.border} strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="type" tick={{ fill: S.txt2, fontSize: 10, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: S.txt3, fontSize: 10, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="days" radius={[4, 4, 0, 0]} name="Days">
                {TYPE_DATA.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '10px' }}>
            {TYPE_DATA.map(d => (
              <span key={d.type} style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '10px', color: S.txt2 }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: d.color, display: 'inline-block' }} />
                {d.type}
              </span>
            ))}
          </div>
        </div>

        {/* Radar — approval rates */}
        <div style={{ ...card }}>
          <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px', marginBottom: '1.5rem' }}>
            APPROVAL RATE & VOLUME BY TYPE
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={TYPE_DATA} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
              <PolarGrid stroke={S.border} />
              <PolarAngleAxis dataKey="type" tick={{ fill: S.txt2, fontSize: 10, fontFamily: "'Space Mono', monospace" }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: S.txt3, fontSize: 9 }} axisLine={false} />
              <Radar name="Approval %" dataKey="approvals" stroke={S.green} fill={S.green} fillOpacity={0.2} strokeWidth={2} />
              <Radar name="Volume ÷100" dataKey="volume"   stroke={S.cyan}  fill={S.cyan}  fillOpacity={0.1} strokeWidth={1.5} />
              <Tooltip content={<CustomTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Seasonal area chart */}
      <div style={{ ...card, marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px' }}>SEASONAL PROCESSING TRENDS</div>
          <div style={{ display: 'flex', gap: '16px', fontSize: '10px' }}>
            {[{ color: S.gold, label: 'Avg Days' }, { color: S.cyan, label: 'Applications' }].map(l => (
              <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: '5px', color: S.txt2 }}>
                <span style={{ width: '18px', height: '2px', background: l.color, display: 'inline-block', borderRadius: '1px' }} />
                {l.label}
              </span>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={SEASONAL_DATA} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="gradDays" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={S.gold} stopOpacity={0.3} />
                <stop offset="95%" stopColor={S.gold} stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradApps" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={S.cyan} stopOpacity={0.2} />
                <stop offset="95%" stopColor={S.cyan} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={S.border} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="m" tick={{ fill: S.txt2, fontSize: 10, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} />
            <YAxis yAxisId="days" tick={{ fill: S.txt3, fontSize: 9, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} />
            <YAxis yAxisId="apps" orientation="right" tick={{ fill: S.txt3, fontSize: 9, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Area yAxisId="days" type="monotone" dataKey="days" stroke={S.gold} fill="url(#gradDays)" strokeWidth={2} name="Avg Days" />
            <Area yAxisId="apps" type="monotone" dataKey="apps" stroke={S.cyan} fill="url(#gradApps)" strokeWidth={1.5} name="Applications" />
          </AreaChart>
        </ResponsiveContainer>
        <div style={{ marginTop: '12px', padding: '12px', background: S.bg3, borderRadius: '8px', fontSize: '11px', color: S.txt2, lineHeight: 1.7 }}>
          <span style={{ color: S.amber }}>⚠ Peak periods:</span> August and December show 40–50% longer processing times.
          Applying in April–May or October–November yields fastest outcomes.
        </div>
      </div>

      {/* Monthly approval rate */}
      <div style={{ ...card }}>
        <div style={{ fontSize: '10px', color: S.txt3, letterSpacing: '2px', marginBottom: '1.5rem' }}>
          MONTHLY APPROVAL RATE TREND (%)
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={SEASONAL_DATA} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
            <CartesianGrid stroke={S.border} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="m" tick={{ fill: S.txt2, fontSize: 10, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} />
            <YAxis domain={[65, 95]} tick={{ fill: S.txt3, fontSize: 9, fontFamily: "'Space Mono', monospace" }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="approvalRate" stroke={S.green} strokeWidth={2} dot={{ fill: S.green, r: 3 }} name="Approval %" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
