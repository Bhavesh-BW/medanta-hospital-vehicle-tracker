/* Vehicle Journey Tracker — full React SPA
   Features:
   - Dynamic Camera Grid: 1 feed (focus), 2 feeds (entry + exit), or all 4 feeds, with add/remove slot controls and auto-enlarge
   - Full-screen mode for any camera stream
   - Continuous background GPU inference across all cameras
   - Vehicle Journeys table with smart camera-wise filtering, status filtering, and pagination (eliminates vertical stretching)
   - Clickable Plate Sightings lightbox modal showing vehicle crop, plate crop, timestamp, and confidence
   - Live KPI cards, engine status, and journey rules
*/
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ReactDOM from 'react-dom/client';
import './theme.css';
import * as XLSX from 'xlsx';

const API = '/api';
const POLL_INTERVAL = 4000;

/* ── Helpers ─────────────────────────────────────────────── */

function ts(ms) {
  if (!ms) return '—';
  const d = new Date(ms);
  return d.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
}

function tsDate(ms) {
  if (!ms) return '—';
  const d = new Date(ms);
  return d.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' }) + ' ' + ts(ms);
}

function dwell(ms) {
  if (!ms && ms !== 0) return '—';
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

function roleColor(role) {
  if (role === 'entry') return 'var(--teal)';
  if (role === 'exit') return 'var(--gold)';
  if (role === 'basement_entry' || role === 'basement') return '#ea580c';
  if (role === 'basement_exit') return '#d97706';
  return 'var(--purple)';
}

const CAMERA_ROLE_OPTIONS = [
  { value: 'entry', label: '🟢 Entry Gate' },
  { value: 'exit', label: '🔵 Exit Gate' },
  { value: 'porch', label: '🟣 Main Porch / Drop-off' },
  { value: 'basement_entry', label: '🟠 Basement Entry' },
  { value: 'basement_exit', label: '🟡 Basement Exit' },
  { value: 'emergency', label: '🔴 Emergency Drop-off' },
  { value: 'checkpoint', label: '📍 General Checkpoint' },
];

const COUNTABLE_CAMERA_ROLES = new Set(['entry', 'exit', 'basement_entry', 'basement_exit', 'basement']);

function normalizeCameraRole(role) {
  const r = String(role || 'checkpoint').toLowerCase();
  return r === 'basement' ? 'basement_entry' : r;
}

function formatRoleLabel(role) {
  if (role === 'basement_entry' || role === 'basement') return 'basement entry';
  if (role === 'basement_exit') return 'basement exit';
  return role || 'checkpoint';
}

function getDirectionMeta(dir) {
  if (dir === 'incoming') return { label: 'INCOMING', pill: 'in' };
  if (dir === 'outgoing') return { label: 'OUTGOING', pill: 'out' };
  if (dir === 'basement_incoming') return { label: 'BASEMENT ENTRY', pill: 'base-in' };
  if (dir === 'basement_outgoing') return { label: 'BASEMENT EXIT', pill: 'base-out' };
  return { label: String(dir || '').toUpperCase(), pill: 'out' };
}

function GateCountingFields({ role, gateGroup, setGateGroup, countEnabled, setCountEnabled }) {
  if (!COUNTABLE_CAMERA_ROLES.has(role)) return null;
  const isBasement = role === 'basement_entry' || role === 'basement_exit' || role === 'basement';
  return (
    <div style={{ padding: '10px 12px', background: '#f8fafd', border: '1px solid #e1e7f0', borderRadius: 8, marginBottom: 12 }}>
      <div className="field" style={{ marginBottom: 8 }}>
        <span>Gate / Location Group (Optional)</span>
        <input
          placeholder={isBasement ? 'e.g. Basement Gate (blank = auto-dedup same-direction)' : 'e.g. Main Gate (blank = auto-dedup all same-role cameras)'}
          value={gateGroup}
          onChange={e => setGateGroup(e.target.value)}
        />
        <small style={{ color: '#64748b', fontSize: 10 }}>
          Cameras with the same Role &amp; Group merge into 1 physical vehicle count.
        </small>
      </div>
      <label className="check">
        <input type="checkbox" checked={countEnabled} onChange={e => setCountEnabled(e.target.checked)} />
        Save counts to Gate History DB (uncheck for Live-Only ANPR)
      </label>
    </div>
  );
}

async function api(path, body) {
  const options = body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {};
  const res = await fetch(`${API}${path}`, options);
  return res.json();
}

/* ── KPI Card ────────────────────────────────────────────── */

function LoadingPill({ show, text = 'Loading…' }) {
  if (!show) return null;
  return (
    <span className="loading-pill" role="status" aria-live="polite">
      <span className="spin-ring" />
      <span>{text}</span>
    </span>
  );
}

function LoadingOverlay({ show, text = 'Loading data…' }) {
  if (!show) return null;
  return (
    <div className="loading-overlay" role="status" aria-live="polite">
      <div className="loading-box">
        <span className="spin-ring lg" />
        <span>{text}</span>
      </div>
    </div>
  );
}

function KpiCard({ label, value, sub, color }) {
  return (
    <div className="kpi" style={{ '--kpi': color }}>
      <span>{label}</span>
      <b>{value ?? '—'}</b>
      {sub && <small>{sub}</small>}
    </div>
  );
}

function BasementInKpiCard({ vehicleCounts, loading = false }) {
  const bIn = vehicleCounts?.basement?.incoming?.car ?? 0;
  const bOut = vehicleCounts?.basement?.outgoing?.car ?? 0;
  return (
    <KpiCard
      label="BASEMENT IN"
      value={vehicleCounts?.basement ? bIn : '—'}
      sub={
        loading ? (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span className="spin-ring" style={{ width: 9, height: 9, borderWidth: 1.5 }} />
            <span>Updating…</span>
          </span>
        ) : (
          vehicleCounts?.basement ? `${bOut} out` : '—'
        )
      }
      color="#ea580c"
    />
  );
}

/* ── Fullscreen Stream Modal ──────────────────────────────── */

function FullscreenModal({ stream, device, worker, feedSource, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!stream) return null;

  const fps = worker?.fps ?? 0;
  const plates = worker?.plates ?? 0;

  return (
    <div className="fullscreen-stage">
      <div className="fullscreen-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 16, fontWeight: 800, color: '#fff' }}>
            📹 {device?.name || stream}
          </span>
          {device?.role && (
            <span className={`badge ${device.role}`} style={{ fontSize: 10, padding: '4px 10px' }}>
              {device.role.toUpperCase()}
            </span>
          )}
          <span style={{ fontSize: 12, fontWeight: 700, color: '#22a06e', background: 'rgba(34,160,110,.15)', padding: '4px 10px', borderRadius: 8 }}>
            🟢 {fps} FPS
          </span>
          {plates > 0 ? (
            <span className="badge on_site" style={{ fontSize: 10 }}>{plates} DETECTED</span>
          ) : (
            <span className="badge on_site" style={{ fontSize: 10 }}>SCANNING</span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <small style={{ color: '#8ca0bc', fontSize: 11 }}>Press [ESC] or click to exit</small>
          <button className="btn primary sm" onClick={onClose} style={{ fontSize: 11, padding: '7px 14px' }}>
            ✕ Exit Fullscreen
          </button>
        </div>
      </div>
      <div className="fullscreen-body" onClick={onClose}>
        <img
          src={`${API}/devices/${stream}/stream.mjpg?mode=${feedSource || 'live'}`}
          alt={device?.name || stream}
          onClick={(e) => e.stopPropagation()}
        />
      </div>
    </div>
  );
}

/* ── Plate Sighting Lightbox Modal ────────────────────────── */

function SightingModal({ sighting, deviceMap, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!sighting) return null;

  const device = deviceMap[sighting.stream];
  const photoUrl = sighting.photo_url ? `${API}${sighting.photo_url}` : null;
  const plateUrl = sighting.id ? `${API}/sightings/${sighting.id}/plate` : null;
  const confPct = sighting.confidence != null ? Math.round(sighting.confidence * 100) : null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" style={{ width: 620, maxWidth: '94vw' }} onClick={e => e.stopPropagation()}>
        {/* Modal Head */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '16px 20px', borderBottom: '1px solid #eef2f7', background: '#fbfcfe'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 20 }}>🚗</span>
            <div>
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 800, color: '#172138' }}>
                Plate Sighting Detection
              </h3>
              <small style={{ color: '#8ca0bc', fontSize: 11 }}>
                Camera Snapshot & License Plate Capture
              </small>
            </div>
          </div>
          <button className="btn sm" onClick={onClose} style={{ borderRadius: '50%', width: 32, height: 32, padding: 0 }}>
            ✕
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '20px' }}>
          {/* Main Vehicle Photo Preview */}
          <div style={{
            background: '#0b1220', borderRadius: 12, overflow: 'hidden',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            minHeight: 240, maxHeight: 380, border: '1px solid #233146', marginBottom: 16
          }}>
            {photoUrl ? (
              <img
                src={photoUrl}
                alt="Detected vehicle"
                style={{ width: '100%', maxHeight: 380, objectFit: 'contain', display: 'block' }}
              />
            ) : (
              <span style={{ color: '#8ca0bc', fontSize: 12 }}>No photo available</span>
            )}
          </div>

          {/* Details & Plate Crop Row */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 16,
            background: '#f8fafd', padding: 14, borderRadius: 12, border: '1px solid #e7ecf4'
          }}>
            {/* Plate Crop */}
            <div style={{
              background: '#000', padding: 8, borderRadius: 8,
              border: '2px solid #ca8c06', display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', minWidth: 150
            }}>
              <span style={{ fontSize: 9, fontWeight: 800, color: '#ca8c06', textTransform: 'uppercase', marginBottom: 4 }}>
                Cropped Plate
              </span>
              {plateUrl ? (
                <img
                  src={plateUrl}
                  alt={sighting.plate}
                  style={{ maxHeight: 46, maxWidth: 170, objectFit: 'contain', display: 'block' }}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              ) : null}
              <div style={{
                marginTop: 6, background: '#fff', color: '#111', fontWeight: 900,
                fontSize: 14, letterSpacing: 1.5, padding: '3px 10px', borderRadius: 4,
                border: '1px solid #222'
              }}>
                {sighting.plate || 'UNKNOWN'}
              </div>
            </div>

            {/* Sighting Metadata */}
            <div style={{ display: 'grid', gap: 6, fontSize: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #eef2f7', paddingBottom: 4 }}>
                <span style={{ color: '#8ca0bc', fontWeight: 600 }}>Detected Time:</span>
                <b style={{ color: '#172138' }}>{tsDate(sighting.ts)}</b>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #eef2f7', paddingBottom: 4 }}>
                <span style={{ color: '#8ca0bc', fontWeight: 600 }}>Camera Stream:</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <b style={{ color: '#172138' }}>{device?.name || sighting.stream}</b>
                  {device?.role && (
                    <span className={`badge ${device.role}`} style={{ fontSize: 8 }}>{device.role}</span>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #eef2f7', paddingBottom: 4 }}>
                <span style={{ color: '#8ca0bc', fontWeight: 600 }}>Confidence:</span>
                <div>
                  {confPct != null ? (
                    <span className={`badge ${confPct >= 80 ? 'on_site' : confPct >= 50 ? 'warn' : 'err'}`}>
                      {confPct}% {confPct >= 80 ? 'High Confidence' : 'Moderate'}
                    </span>
                  ) : '—'}
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', paddingTop: 2 }}>
                <span style={{ color: '#8ca0bc', fontWeight: 600 }}>Record ID:</span>
                <span style={{ fontFamily: 'monospace', fontSize: 11, color: '#5d7089' }}>{sighting.id?.slice(0, 16)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', padding: '12px 20px', borderTop: '1px solid #eef2f7', background: '#fafbfc' }}>
          <button className="btn primary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
}

/* ── Camera Slot (Live MJPEG + Fullscreen + Remove) ───────── */

function CameraSlot({
  slotIndex,
  stream,
  devices,
  activeSlots,
  onChangeSlot,
  onRemoveSlot,
  onFullscreen,
  canRemove,
  engineWorkers,
  videoHeight,
  feedSource
}) {
  const device = devices.find(d => d.stream === stream);
  const worker = stream ? engineWorkers?.[stream] : null;
  const fps = worker?.fps ?? 0;
  const hasDetections = worker?.plates > 0;
  const hasError = worker?.error;

  let borderColor = '#2a3a4f';
  let statusBadge = null;
  if (!stream) {
    borderColor = '#2a3a4f';
  } else if (hasError) {
    borderColor = '#c04848';
    statusBadge = <span className="badge err" style={{ fontSize: '9px' }}>ERROR</span>;
  } else if (hasDetections) {
    borderColor = '#22a06e';
    statusBadge = <span className="badge on_site" style={{ fontSize: '9px' }}>{worker.plates} PLATE{worker.plates !== 1 ? 'S' : ''}</span>;
  } else {
    borderColor = '#22a06e';
    statusBadge = <span className="badge on_site" style={{ fontSize: '9px' }}>SCANNING</span>;
  }

  return (
    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
      {/* Slot header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '6px 0', gap: 6, flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
          <span className="badge" style={{ background: 'var(--primary-500)', color: '#fff', fontSize: '9px', padding: '3px 8px', flexShrink: 0 }}>
            SLOT {slotIndex + 1}
          </span>
          {device && (
            <span style={{ fontSize: 12, fontWeight: 700, color: '#e0e8f0', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              📹 {device.name}
            </span>
          )}
          {device?.role && (
            <span className={`badge ${device.role}`} style={{ fontSize: '8px', padding: '2px 6px', flexShrink: 0 }}>
              {device.role}
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
          {stream && (
            <span style={{ fontSize: 10, color: '#8ca0bc', fontWeight: 600 }}>{fps} FPS</span>
          )}
          {statusBadge}
          {stream && (
            <button
              title="Fullscreen"
              className="btn sm"
              onClick={() => onFullscreen(stream)}
              style={{ padding: '3px 7px', background: '#1c293c', color: '#c3d0e0', border: '1px solid #324765', borderRadius: 6 }}
            >
              ⛶
            </button>
          )}
          {canRemove && (
            <button
              title="Remove this camera from grid view"
              className="btn sm"
              onClick={() => onRemoveSlot(slotIndex)}
              style={{ padding: '3px 7px', background: '#251a24', color: '#e68a8a', border: '1px solid #5a2e3a', borderRadius: 6 }}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Video area */}
      <div style={{
        position: 'relative',
        background: 'linear-gradient(180deg, #101a2c, #0b1220)',
        borderRadius: 10,
        border: `2px solid ${borderColor}`,
        overflow: 'hidden',
        height: videoHeight,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'height 0.2s ease-in-out'
      }}>
        {stream ? (
          feedSource ? (
            <img
              src={`${API}/devices/${stream}/stream.mjpg?mode=${feedSource}`}
              alt={device?.name || stream}
              style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          ) : (
            <div style={{ color: '#5d7089', fontSize: 11, textAlign: 'center', padding: 20 }}>
              Connecting stream...
            </div>
          )
        ) : (
          <div style={{ color: '#5d7089', fontSize: 11, textAlign: 'center', padding: 20 }}>
            No camera assigned to Slot {slotIndex + 1}<br />
            <small>Select a camera from the dropdown below</small>
          </div>
        )}
      </div>

      {/* Camera selector dropdown */}
      <div style={{ marginTop: 6 }}>
        <select
          value={stream || ''}
          onChange={e => onChangeSlot(slotIndex, e.target.value || null)}
          style={{
            width: '100%', padding: '7px 10px', border: '1px solid #2a3a4f',
            borderRadius: 8, background: '#0f1a2c', color: '#c3d0e0',
            fontSize: 11, fontWeight: 700
          }}
        >
          <option value="">— Select Camera —</option>
          {devices.map(d => {
            const inSlot = activeSlots.find((s, idx) => s === d.stream && idx !== slotIndex);
            return (
              <option key={d.stream} value={d.stream}>
                {d.name} {inSlot ? '• In View' : ''}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  );
}

/* ── Add Camera Modal ─────────────────────────────────────── */

function AddCameraModal({ open, onClose, onAdd }) {
  const [url, setUrl] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState('checkpoint');
  const [anpr, setAnpr] = useState(true);
  const [gateGroup, setGateGroup] = useState('');
  const [countEnabled, setCountEnabled] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  if (!open) return null;

  const deriveStream = (rawUrl) => {
    if (!rawUrl || typeof rawUrl !== 'string') return '';
    const trimmed = rawUrl.trim();
    if (!trimmed.includes('/') && !trimmed.includes(':')) {
      return trimmed.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 40);
    }
    const clean = trimmed.split('?')[0].replace(/\/+$/, '');
    const parts = clean.split('/');
    const last = parts[parts.length - 1] || '';
    const cand = last.replace(/[^A-Za-z0-9_-]/g, '').slice(0, 40);
    if (cand) return cand;
    return 'cam_' + Math.abs(clean.split('').reduce((a, b) => ((a << 5) - a + b.charCodeAt(0)) | 0, 0)).toString(36);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return setError('Camera URL is required');
    const stream = deriveStream(url.trim());
    if (!stream) return setError('Please enter a valid Camera URL.');
    setBusy(true);
    setError('');
    try {
      const result = await api('/devices', {
        stream,
        url: url.trim(),
        name: name.trim() || stream,
        role,
        anpr,
        gate_group: gateGroup.trim(),
        count_enabled: countEnabled
      });
      if (result.error) throw new Error(result.error);
      onAdd(result);
      setUrl(''); setName(''); setRole('checkpoint'); setAnpr(true); setGateGroup(''); setCountEnabled(true);
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="card modal-box" style={{ width: 460, maxWidth: '92vw' }} onClick={e => e.stopPropagation()}>
        <div className="head">
          <h3>Register New Camera</h3>
          <button className="btn sm" onClick={onClose}>✕</button>
        </div>
        <div className="body">
          {error && <div className="notice err">{error}</div>}
          <form onSubmit={handleSubmit}>
            <div className="field" style={{ marginBottom: 12 }}>
              <span>Camera URL</span>
              <input placeholder="e.g. rtsp://124.123.89.162:31554/live/mgptz" value={url} onChange={e => setUrl(e.target.value)} />
              <small style={{ color: '#8ca0bc', fontSize: 10 }}>
                RTSP, RTMP, or camera stream URL
              </small>
            </div>
            <div className="field" style={{ marginBottom: 12 }}>
              <span>Display Name</span>
              <input placeholder="e.g. Main Gate Entry" value={name} onChange={e => setName(e.target.value)} />
            </div>
            <div className="form-row" style={{ marginBottom: 12 }}>
              <div className="field" style={{ flex: 1 }}>
                <span>Role / Zone</span>
                <select value={role} onChange={e => setRole(e.target.value)}>
                  {CAMERA_ROLE_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
              <label className="check">
                <input type="checkbox" checked={anpr} onChange={e => setAnpr(e.target.checked)} />
                ANPR Enabled
              </label>
            </div>
            <GateCountingFields
              role={role}
              gateGroup={gateGroup}
              setGateGroup={setGateGroup}
              countEnabled={countEnabled}
              setCountEnabled={setCountEnabled}
            />
            <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 16 }}>
              <button type="button" className="btn" onClick={onClose}>Cancel</button>
              <button type="submit" className="btn primary" disabled={busy}>
                {busy ? 'Adding…' : '+ Add Camera'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

/* ── Edit Camera Modal ────────────────────────────────────── */

function EditCameraModal({ device, open, onClose, onSave, onDelete }) {
  const [name, setName] = useState(device?.name || '');
  const [url, setUrl] = useState(device?.url || '');
  const [role, setRole] = useState(normalizeCameraRole(device?.role));
  const [anpr, setAnpr] = useState(device?.anpr ?? true);
  const [gateGroup, setGateGroup] = useState(device?.gate_group || '');
  const [countEnabled, setCountEnabled] = useState(device?.count_enabled ?? true);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (device) {
      setName(device.name);
      setUrl(device.url || '');
      setRole(normalizeCameraRole(device.role));
      setAnpr(device.anpr);
      setGateGroup(device.gate_group || '');
      setCountEnabled(device.count_enabled ?? true);
    }
  }, [device]);

  if (!open || !device) return null;

  const handleSave = async () => {
    setBusy(true);
    try {
      await api('/devices', {
        stream: device.stream,
        url: url.trim(),
        name,
        role,
        anpr,
        gate_group: gateGroup.trim(),
        count_enabled: countEnabled
      });
      onSave();
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm(`Remove camera "${device.name}"?`)) return;
    setBusy(true);
    try {
      await api('/devices/delete', { stream: device.stream });
      onDelete();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="card modal-box" style={{ width: 440, maxWidth: '92vw' }} onClick={e => e.stopPropagation()}>
        <div className="head">
          <h3>Edit Camera — {device.name}</h3>
          <button className="btn sm" onClick={onClose}>✕</button>
        </div>
        <div className="body">
          <div className="field" style={{ marginBottom: 12 }}>
            <span>Camera URL</span>
            <input value={url} onChange={e => setUrl(e.target.value)} placeholder="rtsp://..." />
          </div>
          <div className="field" style={{ marginBottom: 12 }}>
            <span>Display Name</span>
            <input value={name} onChange={e => setName(e.target.value)} />
          </div>
          <div className="form-row" style={{ marginBottom: 12 }}>
            <div className="field" style={{ flex: 1 }}>
              <span>Role / Zone</span>
              <select value={role} onChange={e => setRole(e.target.value)}>
                {CAMERA_ROLE_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
            <label className="check">
              <input type="checkbox" checked={anpr} onChange={e => setAnpr(e.target.checked)} />
              ANPR Enabled
            </label>
          </div>
          <GateCountingFields
            role={role}
            gateGroup={gateGroup}
            setGateGroup={setGateGroup}
            countEnabled={countEnabled}
            setCountEnabled={setCountEnabled}
          />
          <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', marginTop: 16 }}>
            <button className="btn danger" onClick={handleDelete} disabled={busy}>Remove</button>
            <button className="btn" onClick={onClose}>Cancel</button>
            <button className="btn primary" onClick={handleSave} disabled={busy}>
              {busy ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Registered Cameras Panel ────────────────────────────── */

function RegisteredCameras({ devices, activeSlots, onToggleViewSlot, onEditCamera, onOpenAdd }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <b style={{ fontSize: 13, color: '#26334a' }}>Registered Cameras ({devices.length})</b>
        <button className="btn sm primary" onClick={onOpenAdd}>+ Add</button>
      </div>
      <div className="devices">
        {devices.map(d => {
          const slotIdx = activeSlots.indexOf(d.stream);
          const inView = slotIdx !== -1;
          return (
            <div key={d.stream} className={`device ${d.anpr ? 'live-dot' : ''}`}>
              <div className="cam">📹</div>
              <div className="info">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, minWidth: 0 }}>
                  <b title={d.name}>{d.name}</b>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4, flex: 'none' }}>
                    <button
                      className="btn sm"
                      onClick={() => onToggleViewSlot(d.stream)}
                      style={{
                        fontSize: 10,
                        padding: '3px 8px',
                        background: inView ? '#f0f4f8' : '#e3f6ef',
                        color: inView ? '#5d7089' : '#17835c',
                        borderColor: inView ? '#dfe6f0' : '#bfe6d4'
                      }}
                    >
                      {inView ? 'Hide' : 'Show'}
                    </button>
                    <button className="btn sm" style={{ padding: '3px 7px' }} onClick={() => onEditCamera(d)}>✏️</button>
                  </div>
                </div>
                <small style={{ fontSize: 9, color: '#8ca0bc', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', display: 'block' }} title={d.url || d.stream}>
                  {d.url || d.stream}
                </small>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap', marginTop: 1 }}>
                  <span className={`badge ${d.role}`} style={{ fontSize: 8 }}>{formatRoleLabel(d.role)}</span>
                  {COUNTABLE_CAMERA_ROLES.has(d.role) && d.gate_group && (
                    <span className="badge" style={{ background: '#e0e7ff', color: '#3730a3', fontSize: 8 }} title={`Gate Group: ${d.gate_group}`}>
                      {d.gate_group}
                    </span>
                  )}
                  {COUNTABLE_CAMERA_ROLES.has(d.role) && d.count_enabled === false && (
                    <span className="badge" style={{ background: '#fef3c7', color: '#92400e', fontSize: 8 }} title="Excluded from Gate DB counts (Live-Only)">
                      Live Only
                    </span>
                  )}
                  {inView && (
                    <span className="badge" style={{ background: 'var(--primary-100)', color: 'var(--primary-600)', fontSize: 8 }}>
                      Slot {slotIdx + 1}
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        {devices.length === 0 && <div className="empty">No cameras registered. Click + Add to get started.</div>}
      </div>
    </div>
  );
}

/* ── Journey Flow (Vehicle Path across Cameras) ──────────── */

function JourneyFlow({ journey, expanded, onToggle, onOpenSightingPhoto }) {
  if (!journey) return null;
  const cameras = journey.cameras || [];
  const vtypeKey = String(journey.vehicle_type || 'car').toLowerCase();
  const vtMeta = VEHICLE_TYPE_META.find(v => v.key === vtypeKey) || VEHICLE_TYPE_META[0];

  return (
    <>
      <tr className="expandable clickable-row" onClick={onToggle}>
        <td><span className="plate">{journey.plate}</span></td>
        <td>
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 8px',
            borderRadius: 6, background: vtMeta.bg, color: vtMeta.color,
            fontSize: 10, fontWeight: 800, whiteSpace: 'nowrap'
          }}>
            <span>{vtMeta.icon}</span>
            <span>{vtMeta.singular}</span>
          </span>
        </td>
        <td>
          <span className={`badge ${journey.status}`}>
            {journey.status === 'on_site' ? '🟢 ON SITE' : '🔵 EXITED'}
          </span>
        </td>
        <td>{journey.entry?.device || '—'}</td>
        <td style={{ whiteSpace: 'nowrap' }}>{ts(journey.first_at)}</td>
        <td>
          {journey.exit ? (
            journey.exit_confirmed === false
              ? <span className="muted" style={{ fontSize: 11 }}>{journey.exit.device}</span>
              : journey.exit.device
          ) : '—'}
        </td>
        <td style={{ whiteSpace: 'nowrap' }}>{journey.exit?.at ? ts(journey.exit.at) : '—'}</td>
        <td><b>{dwell(journey.dwell_ms)}</b></td>
        <td>
          <div className="camchips">
            {cameras.map((c, i) => (
              <span key={i} className={`camchip ${c.role}`}>{c.name}</span>
            ))}
          </div>
        </td>
        <td style={{ textAlign: 'center' }}><b>{journey.reads}</b></td>
      </tr>
      {expanded && (
        <tr className="journey-detail">
          <td colSpan={10} style={{ padding: '12px 16px', background: '#f8fafd' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <div style={{ fontSize: 12, fontWeight: 800, color: '#26334a', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                <span>🚗 Vehicle Travel Timeline & Correlation — <span className="plate">{journey.plate}</span></span>
                <span style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 8px',
                  borderRadius: 6, background: vtMeta.bg, color: vtMeta.color,
                  fontSize: 10, fontWeight: 800
                }}>
                  {vtMeta.icon} {vtMeta.singular}
                </span>
              </div>
              <span className={`badge ${journey.status}`}>
                {journey.status === 'on_site' ? '🟢 Currently On Hospital Grounds' : '🔵 Journey Complete / Exited'}
              </span>
            </div>

            {/* Travel nodes */}
            <div className="jt">
              {cameras.map((cam, i) => (
                <React.Fragment key={cam.stream + i}>
                  {i > 0 && <div className="jt-arrow">→</div>}
                  <div className="jt-node" style={{ '--role-c': roleColor(cam.role) }}>
                    <div className="role">{cam.role}</div>
                    <b>{cam.name}</b>
                    <small>{ts(cam.first)} — {ts(cam.last)}</small>
                    <div className="reads">
                      {cam.reads} read{cam.reads !== 1 ? 's' : ''} • conf {Math.round((cam.confidence || 0) * 100)}%
                    </div>
                  </div>
                </React.Fragment>
              ))}
            </div>

            {/* Thumbnail captures */}
            <div className="jt-imgs" style={{ marginTop: 10 }}>
              {journey.entry?.photo_url && (
                <figure
                  className="clickable-thumb"
                  onClick={(e) => {
                    e.stopPropagation();
                    onOpenSightingPhoto({
                      id: journey.entry.id,
                      photo_url: journey.entry.photo_url,
                      plate: journey.plate,
                      stream: journey.entry.stream || journey.entry.device,
                      ts: journey.first_at,
                      confidence: journey.entry.confidence
                    });
                  }}
                  title="Click to view full image"
                  style={{ cursor: 'pointer' }}
                >
                  <img src={`${API}${journey.entry.photo_url}`} alt="Entry Sighting" />
                  <figcaption>🔍 Entry: {journey.entry.device} ({ts(journey.first_at)})</figcaption>
                </figure>
              )}
              {journey.exit?.photo_url && (
                <figure
                  className="clickable-thumb"
                  onClick={(e) => {
                    e.stopPropagation();
                    onOpenSightingPhoto({
                      id: journey.exit.id,
                      photo_url: journey.exit.photo_url,
                      plate: journey.plate,
                      stream: journey.exit.stream || journey.exit.device,
                      ts: journey.exit.at,
                      confidence: journey.exit.confidence
                    });
                  }}
                  title="Click to view full image"
                  style={{ cursor: 'pointer' }}
                >
                  <img src={`${API}${journey.exit.photo_url}`} alt="Exit Sighting" />
                  <figcaption>🔍 Exit: {journey.exit.device} ({ts(journey.exit.at)})</figcaption>
                </figure>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

/* ── Dynamic Device Zone Classification Helper ──────────────── */

function getDeviceZone(device) {
  const name = (device?.name || '').toLowerCase();
  const role = (device?.role || '').toLowerCase();
  if (role === 'entry') return { key: 'entry', label: 'Entry Gate', icon: '🟢', badgeBg: '#dcfce7', badgeColor: '#15803d', barColor: '#16a34a' };
  if (role === 'exit') return { key: 'exit', label: 'Exit Gate', icon: '🔵', badgeBg: '#dbeafe', badgeColor: '#1d4ed8', barColor: '#2563eb' };
  if (role === 'porch' || name.includes('porch') || name.includes('drop-off')) {
    return { key: 'porch', label: 'Porch', icon: '🟣', badgeBg: '#f3e8ff', badgeColor: '#7e22ce', barColor: '#9333ea' };
  }
  if (role === 'basement_exit' || name.includes('basement exit')) {
    return { key: 'basement_exit', label: 'Basement Exit', icon: '🟡', badgeBg: '#fef3c7', badgeColor: '#a16207', barColor: '#d97706' };
  }
  if (role === 'basement_entry' || role === 'basement' || name.includes('basement') || name.includes('ramp')) {
    return { key: 'basement_entry', label: 'Basement Entry', icon: '🟠', badgeBg: '#ffedd5', badgeColor: '#c2410c', barColor: '#ea580c' };
  }
  if (role === 'parking' || name.includes('parking')) {
    return { key: 'parking', label: 'Parking', icon: '🅿️', badgeBg: '#e0f2fe', badgeColor: '#0369a1', barColor: '#0284c7' };
  }
  if (role === 'emergency' || name.includes('emergency') || name.includes('ambulance')) {
    return { key: 'emergency', label: 'Emergency', icon: '🔴', badgeBg: '#fee2e2', badgeColor: '#b91c1c', barColor: '#dc2626' };
  }
  return { key: 'checkpoint', label: 'Internal', icon: '📍', badgeBg: '#f1f5f9', badgeColor: '#475569', barColor: '#6366f1' };
}

/* ── Gate Traffic & Camera Flow Analysis ──────────────────── */

function GateAnalyticsCard({ analytics, devices, timeFilter, setTimeFilter, loading = false }) {
  const perCamera = analytics?.per_camera || {};

  const entryDevices = (devices || []).filter(d => d.role === 'entry');
  const exitDevices = (devices || []).filter(d => d.role === 'exit');
  // All internal devices: Porch, Basement, Parking, Checkpoint, Drop-off, etc.
  const internalDevices = (devices || []).filter(d => d.role !== 'entry' && d.role !== 'exit');

  const totalEntries = entryDevices.reduce((sum, d) => sum + (perCamera[d.stream] || 0), 0) || (analytics?.vehicles_today || 0);
  const cameraExits = exitDevices.reduce((sum, d) => sum + (perCamera[d.stream] || 0), 0);
  const onSiteCount = analytics?.on_site ?? 0;
  const totalExited = analytics?.exited ?? (totalEntries > onSiteCount ? totalEntries - onSiteCount : cameraExits);
  const inferredExits = Math.max(0, totalExited - cameraExits);
  const totalInternal = internalDevices.reduce((sum, d) => sum + (perCamera[d.stream] || 0), 0);
  const clearanceRate = totalEntries > 0 ? Math.min(100, Math.round((totalExited / totalEntries) * 100)) : 0;

  return (
    <div className="card">
      {/* Head */}
      <div className="head" style={{ padding: '14px 18px', flexWrap: 'wrap', gap: 10 }}>
        <div>
          <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
            <span>📊</span> Gate Traffic & Camera Flow Analysis
            <LoadingPill show={loading} text="Loading…" />
          </h3>
          <small style={{ color: '#8ca0bc', marginTop: 2, display: 'block' }}>
            Real-time vehicle passage volume across entry gates, internal zones, and exit checkpoints
          </small>
        </div>

        {/* Quick Time Range Presets */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
          {[
            { key: '1h', label: '1h' },
            { key: '6h', label: '6h' },
            { key: 'today', label: 'Today' },
            { key: '24h', label: '24h' },
            { key: '7d', label: '7 Days' },
          ].map(btn => (
            <button
              key={btn.key}
              onClick={() => setTimeFilter && setTimeFilter(btn.key)}
              style={{
                padding: '4px 10px',
                fontSize: 11,
                fontWeight: 700,
                borderRadius: 6,
                border: timeFilter === btn.key ? '1px solid #0369a1' : '1px solid #cbd5e1',
                background: timeFilter === btn.key ? '#0284c7' : '#fff',
                color: timeFilter === btn.key ? '#fff' : '#475569',
                cursor: 'pointer'
              }}
            >
              {btn.label}
            </button>
          ))}
          <span className="badge" style={{ background: '#e0f2fe', color: '#0369a1', fontWeight: 700, fontSize: 11 }}>
            ⚡ Low-Latency GPU Telemetry
          </span>
        </div>
      </div>

      <div className="body loading-wrap" style={{ padding: '18px' }}>
        <LoadingOverlay show={loading} text="Updating gate analysis…" />
        {/* KPI Mini Summary Tiles */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 14, marginBottom: 20 }}>
          <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 10, padding: '12px 16px' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#166534', textTransform: 'uppercase', marginBottom: 4 }}>
              🟢 Inflow (Entry Gates)
            </div>
            <div style={{ fontSize: 24, fontWeight: 900, color: '#14532d' }}>
              {totalEntries.toLocaleString()}
            </div>
            <div style={{ fontSize: 11, color: '#15803d', marginTop: 2 }}>
              Vehicles entered campus
            </div>
          </div>

          <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 10, padding: '12px 16px' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#1e40af', textTransform: 'uppercase', marginBottom: 4 }}>
              🔵 Total Departed
            </div>
            <div style={{ fontSize: 24, fontWeight: 900, color: '#1e3a8a' }}>
              {totalExited.toLocaleString()}
            </div>
            <div style={{ fontSize: 11, color: '#2563eb', marginTop: 2 }}>
              {cameraExits} camera verified • {inferredExits} grace timeout
            </div>
          </div>

          <div style={{ background: '#fefce8', border: '1px solid #fef08a', borderRadius: 10, padding: '12px 16px' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#854d0e', textTransform: 'uppercase', marginBottom: 4 }}>
              🟡 Currently Inside
            </div>
            <div style={{ fontSize: 24, fontWeight: 900, color: '#713f12' }}>
              {onSiteCount.toLocaleString()}
            </div>
            <div style={{ fontSize: 11, color: '#a16207', marginTop: 2 }}>
              Active on campus (seen &lt; 45m)
            </div>
          </div>

          <div style={{ background: '#faf5ff', border: '1px solid #e9d5ff', borderRadius: 10, padding: '12px 16px' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#6b21a8', textTransform: 'uppercase', marginBottom: 4 }}>
              ⏱ Clearance Rate
            </div>
            <div style={{ fontSize: 24, fontWeight: 900, color: '#581c87' }}>
              {clearanceRate}%
            </div>
            <div style={{ fontSize: 11, color: '#7e22ce', marginTop: 2 }}>
              Exited vs entered ratio
            </div>
          </div>
        </div>

        {/* Gate Breakdown Columns: Entry vs Checkpoints/Internal Zones vs Exit */}
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(auto-fit, minmax(${internalDevices.length > 0 ? '240px' : '280px'}, 1fr))`, gap: 18 }}>
          {/* Entry Gates Column */}
          <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 14 }}>🟢</span>
                <span style={{ fontWeight: 800, fontSize: 13, color: '#0f172a' }}>Entry Checkpoints</span>
              </div>
              <span className="badge" style={{ background: '#dcfce7', color: '#15803d', fontWeight: 700, fontSize: 11 }}>
                {totalEntries.toLocaleString()} Total
              </span>
            </div>

            <div style={{ display: 'grid', gap: 12 }}>
              {entryDevices.map(d => {
                const count = perCamera[d.stream] || 0;
                const pct = totalEntries > 0 ? Math.round((count / totalEntries) * 100) : 0;
                return (
                  <div key={d.stream} style={{ background: '#f8fafc', padding: '10px 14px', borderRadius: 8, border: '1px solid #f1f5f9' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <span style={{ fontWeight: 700, fontSize: 12, color: '#1e293b' }}>{d.name}</span>
                      <span style={{ fontSize: 12, fontWeight: 800, color: '#0f172a' }}>
                        {count.toLocaleString()} <span style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>({pct}%)</span>
                      </span>
                    </div>
                    <div style={{ height: 7, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${pct}%`, height: '100%', background: '#16a34a', borderRadius: 4, transition: 'width 0.4s ease' }} />
                    </div>
                  </div>
                );
              })}
              {entryDevices.length === 0 && (
                <div style={{ color: '#94a3b8', fontSize: 12, textAlign: 'center', padding: 12 }}>No entry cameras registered</div>
              )}
            </div>
          </div>

          {/* Internal Zones & Checkpoints Column (Dynamically surfaces and handles Porch, Basement, Parking, etc.) */}
          {internalDevices.length > 0 && (
            <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 14 }}>🏢</span>
                  <span style={{ fontWeight: 800, fontSize: 13, color: '#0f172a' }}>Internal Zones & Checkpoints</span>
                </div>
                <span className="badge" style={{ background: '#f3e8ff', color: '#7e22ce', fontWeight: 700, fontSize: 11 }}>
                  {totalInternal.toLocaleString()} Total
                </span>
              </div>

              <div style={{ display: 'grid', gap: 12 }}>
                {internalDevices.map(d => {
                  const zone = getDeviceZone(d);
                  const count = perCamera[d.stream] || 0;
                  const pct = totalEntries > 0 ? Math.round((count / totalEntries) * 100) : 0;
                  return (
                    <div key={d.stream} style={{ background: '#f8fafc', padding: '10px 14px', borderRadius: 8, border: '1px solid #f1f5f9' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span style={{ fontSize: 9, fontWeight: 800, background: zone.badgeBg, color: zone.badgeColor, padding: '2px 6px', borderRadius: 4, textTransform: 'uppercase' }}>
                            {zone.icon} {zone.label}
                          </span>
                          <span style={{ fontWeight: 700, fontSize: 12, color: '#1e293b' }}>{d.name}</span>
                        </div>
                        <span style={{ fontSize: 12, fontWeight: 800, color: '#0f172a' }}>
                          {count.toLocaleString()} <span style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>({pct}% of entries)</span>
                        </span>
                      </div>
                      <div style={{ height: 7, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min(100, pct)}%`, height: '100%', background: zone.barColor, borderRadius: 4, transition: 'width 0.4s ease' }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Exit Gates Column */}
          <div style={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 14 }}>🔵</span>
                <span style={{ fontWeight: 800, fontSize: 13, color: '#0f172a' }}>Exit Checkpoints</span>
              </div>
              <span className="badge" style={{ background: '#dbeafe', color: '#1d4ed8', fontWeight: 700, fontSize: 11 }}>
                {cameraExits.toLocaleString()} Camera Verified
              </span>
            </div>

            <div style={{ display: 'grid', gap: 12 }}>
              {exitDevices.map(d => {
                const count = perCamera[d.stream] || 0;
                const pct = cameraExits > 0 ? Math.round((count / cameraExits) * 100) : 0;
                return (
                  <div key={d.stream} style={{ background: '#f8fafc', padding: '10px 14px', borderRadius: 8, border: '1px solid #f1f5f9' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <span style={{ fontWeight: 700, fontSize: 12, color: '#1e293b' }}>{d.name}</span>
                      <span style={{ fontSize: 12, fontWeight: 800, color: '#0f172a' }}>
                        {count.toLocaleString()} <span style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>({pct}%)</span>
                      </span>
                    </div>
                    <div style={{ height: 7, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
                      <div style={{ width: `${pct}%`, height: '100%', background: '#2563eb', borderRadius: 4, transition: 'width 0.4s ease' }} />
                    </div>
                  </div>
                );
              })}
              {exitDevices.length === 0 && (
                <div style={{ color: '#94a3b8', fontSize: 12, textAlign: 'center', padding: 12 }}>No exit cameras registered</div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Inflow Volume Timeline (Hourly / Daily Pulse Card) ──────── */

function InflowVolumeTimelineCard({ analytics, timeFilter, loading = false }) {
  const [timelineMode, setTimelineMode] = useState('hourly'); // 'hourly' | 'daily'

  const hourly = analytics?.hourly_entries || [];
  const dailyEntries = useMemo(() => {
    const map = {};
    (analytics?.hourly_entries || []).forEach(h => {
      const d = new Date(h.bucket);
      const dayLabel = d.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short' });
      map[dayLabel] = (map[dayLabel] || 0) + h.count;
    });
    return Object.entries(map).map(([dayLabel, count]) => ({ dayLabel, count }));
  }, [analytics?.hourly_entries]);

  const isDaily = timelineMode === 'daily' || timeFilter === '7d';
  const timelineItems = isDaily ? dailyEntries : hourly;

  const maxPeak = Math.max(...timelineItems.map(item => item.count), 1);
  const windowTotal = timelineItems.reduce((acc, item) => acc + (item.count || 0), 0);
  const windowAvg = timelineItems.length > 0 ? Math.round(windowTotal / timelineItems.length) : 0;
  const normalizedBars = timelineItems.map(item => {
    const label = isDaily
      ? item.dayLabel
      : new Date(item.bucket).toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: 'numeric', hour12: true });
    return {
      key: isDaily ? item.dayLabel : item.bucket,
      label,
      count: item.count || 0
    };
  });

  if (normalizedBars.length === 0 && !loading) return null;

  return (
    <div className="tl-card loading-wrap" style={{ marginTop: 0, borderRadius: 16, padding: '16px 18px' }}>
      <LoadingOverlay show={loading} text="Updating timeline…" />
      <div className="tl-head">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 800, fontSize: 13, color: '#0f172a', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            📈 {isDaily ? 'Daily Traffic Volume' : 'Inflow Volume Timeline'}
            <span style={{ fontSize: 10, fontWeight: 600, color: '#64748b' }}>
              ({isDaily ? 'Daily Trend' : 'Hourly Pulse'})
            </span>
          </span>
          <LoadingPill show={loading} text="Loading…" />
          {/* Granularity Switcher: Hourly vs Daily */}
          <div style={{ display: 'inline-flex', background: '#e2e8f0', borderRadius: 7, padding: 2 }}>
            <button
              type="button"
              onClick={() => setTimelineMode('hourly')}
              style={{
                padding: '3px 9px', fontSize: 10, fontWeight: 700, border: 'none', borderRadius: 5,
                background: !isDaily ? '#fff' : 'transparent',
                color: !isDaily ? '#0f172a' : '#64748b', cursor: 'pointer',
                boxShadow: !isDaily ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
              }}
            >
              ⏱ Hourly
            </button>
            <button
              type="button"
              onClick={() => setTimelineMode('daily')}
              style={{
                padding: '3px 9px', fontSize: 10, fontWeight: 700, border: 'none', borderRadius: 5,
                background: isDaily ? '#fff' : 'transparent',
                color: isDaily ? '#0f172a' : '#64748b', cursor: 'pointer',
                boxShadow: isDaily ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
              }}
            >
              📅 Daily
            </button>
          </div>
        </div>

        {/* Metrics Summary Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span className="tl-pill peak">
            🔥 Peak: <b>{maxPeak.toLocaleString()}</b> {isDaily ? '/day' : '/hr'}
          </span>
          <span className="tl-pill">
            Avg: <b>{windowAvg.toLocaleString()}</b>
          </span>
          <span className={`tl-pill total ${isDaily ? 'daily' : ''}`}>
            Total: <b>{windowTotal.toLocaleString()}</b>
          </span>
        </div>
      </div>

      {/* Chart Plot Canvas */}
      <div className="tl-plot-wrap">
        <div className="tl-yaxis">
          <span>{maxPeak}</span>
          <span>{Math.round(maxPeak / 2)}</span>
          <span>0</span>
        </div>

        <div className="tl-canvas">
          <div className="tl-canvas-inner" style={{ minWidth: normalizedBars.length > 8 ? `${normalizedBars.length * 64}px` : '100%' }}>
            <div className="tl-gridlines">
              <div className="tl-gridline" />
              <div className="tl-gridline" />
              <div className="tl-gridline base" />
            </div>

            {normalizedBars.length > 1 && windowAvg > 0 && (
              <div
                className="tl-avg-line"
                style={{
                  top: `${24 + Math.round(Math.max(0, Math.min(1, 1 - windowAvg / maxPeak)) * 108)}px`
                }}
              >
                <span className="tl-avg-tag">avg {windowAvg}</span>
              </div>
            )}

            <div className="tl-bars">
              {normalizedBars.map(bar => {
                const barPx = bar.count > 0 ? Math.max(6, Math.round((bar.count / maxPeak) * 108)) : 3;
                const sharePct = windowTotal > 0 ? Math.round((bar.count / windowTotal) * 100) : 0;
                const isPeak = bar.count === maxPeak && maxPeak > 0;
                const modeClass = isDaily ? 'daily' : 'hourly';
                return (
                  <div key={bar.key} className="tl-col" title={`${bar.label}: ${bar.count.toLocaleString()} vehicles (${sharePct}% of window)`}>
                    <div className="tl-track">
                      <span className={`tl-val ${isPeak ? 'peak' : ''}`}>
                        {isPeak ? '★ ' : ''}{bar.count.toLocaleString()}
                      </span>
                      <div
                        className={`tl-bar ${modeClass} ${isPeak ? 'peak' : ''}`}
                        style={{ height: `${barPx}px` }}
                      />
                    </div>
                    <div className="tl-xlab">
                      <b>{bar.label}</b>
                      <small>{sharePct}%</small>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Security Gate Vehicle Counts (Incoming/Outgoing DB + Remaining Live) ── */

const VEHICLE_TYPE_META = [
  { key: 'car', label: 'Cars', singular: 'Car', icon: '🚗', color: '#235496', bg: '#eaf1fa' },
  { key: 'bike', label: 'Bikes', singular: 'Bike', icon: '🏍️', color: '#6932b5', bg: '#f3effc' },
  { key: 'auto', label: 'Autos', singular: 'Auto', icon: '🛺', color: '#147a6d', bg: '#e7f7f4' },
  { key: 'ambulance', label: 'Ambulances', singular: 'Ambulance', icon: '🚑', color: '#b82323', bg: '#feebeb' },
];

const BASEMENT_VEHICLE_TYPE_META = VEHICLE_TYPE_META.filter(vt => vt.key === 'car');

function VehicleCountMatrixCard({
  title,
  subtitle,
  inLabel,
  outLabel,
  inPillClass = 'in',
  outPillClass = 'out',
  totalLabel,
  allTimeLabel,
  vehicleTypes = VEHICLE_TYPE_META,
  incoming,
  outgoing,
  allTimeIn = 0,
  allTimeOut = 0,
  range,
  setRange,
  customStart,
  setCustomStart,
  customEnd,
  setCustomEnd,
  loading = false,
  onOpenHistory
}) {
  const isFullTypes = vehicleTypes.length === VEHICLE_TYPE_META.length;
  const inTotal = isFullTypes
    ? (incoming.total || 0)
    : vehicleTypes.reduce((sum, vt) => sum + (incoming[vt.key] || 0), 0);
  const outTotal = isFullTypes
    ? (outgoing.total || 0)
    : vehicleTypes.reduce((sum, vt) => sum + (outgoing[vt.key] || 0), 0);
  const totalNet = Math.max(0, inTotal - outTotal);

  return (
    <div className="card">
      <div className="head" style={{ flexWrap: 'wrap', gap: 8 }}>
        <div>
          <h3>{title}</h3>
          <small>{subtitle}</small>
        </div>
        <LoadingPill show={loading} text="Loading…" />
      </div>

      <div className="body" style={{ padding: '12px 16px' }}>
        {/* Historical Range Quick Switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap', marginBottom: 10 }}>
          {[
            { k: '1h', label: '1h' },
            { k: 'today', label: 'Today' },
            { k: '24h', label: '24h' },
            { k: '7d', label: '7d' },
            { k: 'all', label: 'All Time' },
            { k: 'custom', label: '📅 Custom' },
          ].map(item => (
            <button
              key={item.k}
              type="button"
              className={`vc-range-btn ${range === item.k ? 'active' : ''}`}
              onClick={() => setRange(item.k)}
            >
              {item.label}
            </button>
          ))}
        </div>

        {range === 'custom' && (
          <div style={{ display: 'grid', gap: 6, padding: 8, background: '#f8fafd', border: '1px solid #e1e7f0', borderRadius: 8, marginBottom: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: '#5d7089' }}>From:</span>
              <input
                type="datetime-local"
                value={customStart}
                onChange={e => setCustomStart(e.target.value)}
                style={{ flex: 1, fontSize: 10, padding: '3px 6px', border: '1px solid #dfe6f0', borderRadius: 5 }}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6 }}>
              <span style={{ fontSize: 10, fontWeight: 700, color: '#5d7089' }}>To:</span>
              <input
                type="datetime-local"
                value={customEnd}
                onChange={e => setCustomEnd(e.target.value)}
                style={{ flex: 1, fontSize: 10, padding: '3px 6px', border: '1px solid #dfe6f0', borderRadius: 5 }}
              />
            </div>
          </div>
        )}

        <div className="loading-wrap" style={{ borderRadius: 8 }}>
          <LoadingOverlay show={loading} text="Updating counts…" />
          {/* Incoming vs Outgoing Breakdown Table */}
          <table className="vc-matrix">
            <thead>
              <tr>
                <th>Vehicle Type</th>
                <th className="num">{inLabel}</th>
                <th className="num">{outLabel}</th>
                <th className="num">Net</th>
              </tr>
            </thead>
            <tbody>
              {vehicleTypes.map(vt => {
                const inCnt = incoming[vt.key] || 0;
                const outCnt = outgoing[vt.key] || 0;
                const net = Math.max(0, inCnt - outCnt);
                return (
                  <tr key={vt.key} style={vt.key === 'ambulance' ? { background: '#fffafa' } : undefined}>
                    <td>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                        <span>{vt.icon}</span>
                        <span style={{ fontWeight: 700, color: vt.key === 'ambulance' ? '#991b1b' : '#1e293b' }}>
                          {vt.label}
                        </span>
                      </span>
                    </td>
                    <td className="num"><span className={`vc-pill ${inPillClass}`}>{inCnt}</span></td>
                    <td className="num"><span className={`vc-pill ${outPillClass}`}>{outCnt}</span></td>
                    <td className="num">
                      <span className={`vc-pill ${net > 0 ? 'net-pos' : 'net-zero'}`}>
                        {net > 0 ? `+${net}` : net}
                      </span>
                    </td>
                  </tr>
                );
              })}
              <tr style={{ background: '#f8fafc' }}>
                <td style={{ fontWeight: 800, color: '#0f172a' }}>{totalLabel}</td>
                <td className="num"><b style={{ color: inPillClass === 'base-in' ? '#c2410c' : '#15803d', fontSize: 12 }}>{inTotal}</b></td>
                <td className="num"><b style={{ color: outPillClass === 'base-out' ? '#a16207' : '#1d4ed8', fontSize: 12 }}>{outTotal}</b></td>
                <td className="num"><b style={{ color: '#334155', fontSize: 12 }}>{totalNet > 0 ? `+${totalNet}` : totalNet}</b></td>
              </tr>
            </tbody>
          </table>

          {/* Footer: All-Time DB Summary + History Modal Trigger */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            gap: 8, marginTop: 12, paddingTop: 10, borderTop: '1px solid #eef2f7', flexWrap: 'wrap'
          }}>
            <small style={{ fontSize: 10, color: '#64748b', fontWeight: 600 }}>
              {allTimeLabel}: <b>{allTimeIn}</b> in • <b>{allTimeOut}</b> out
            </small>
            <button type="button" className="btn sm primary" onClick={onOpenHistory} style={{ padding: '5px 10px' }}>
              📜 Historical Records
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function SecurityVehicleCountsCard({
  data, range, setRange, customStart, setCustomStart, customEnd, setCustomEnd, loading = false, onOpenHistory
}) {
  const emptyBucket = { car: 0, bike: 0, auto: 0, ambulance: 0, total: 0 };
  const incoming = data?.incoming || emptyBucket;
  const outgoing = data?.outgoing || emptyBucket;
  const allTime = data?.all_time || {};
  const basementIn = data?.basement?.incoming || emptyBucket;
  const basementOut = data?.basement?.outgoing || emptyBucket;
  const basementAll = data?.basement?.all_time || {};

  return (
    <>
      {/* 1. Saved Main Gate Incoming & Outgoing Counts Card */}
      <VehicleCountMatrixCard
        title="🛡️ Gate Vehicle Counts"
        subtitle="Incoming & Outgoing • Saved in Database"
        inLabel="🟢 Incoming"
        outLabel="🔵 Outgoing"
        inPillClass="in"
        outPillClass="out"
        totalLabel="Total Gate Flow"
        allTimeLabel="All-time DB"
        vehicleTypes={VEHICLE_TYPE_META}
        incoming={incoming}
        outgoing={outgoing}
        allTimeIn={allTime.incoming?.total || 0}
        allTimeOut={allTime.outgoing?.total || 0}
        range={range}
        setRange={setRange}
        customStart={customStart}
        setCustomStart={setCustomStart}
        customEnd={customEnd}
        setCustomEnd={setCustomEnd}
        loading={loading}
        onOpenHistory={() => onOpenHistory('main')}
      />

      {/* 2. Separate Basement Entry & Basement Exit Counts Card (Cars Only) */}
      <VehicleCountMatrixCard
        title="🅿️ Gate Vehicle Counts — Basement"
        subtitle="Basement Entry & Basement Exit • Cars Only • Saved in Database"
        inLabel="🟠 Basement Entry"
        outLabel="🟡 Basement Exit"
        inPillClass="base-in"
        outPillClass="base-out"
        totalLabel="Total Basement Flow"
        allTimeLabel="All-time Basement DB"
        vehicleTypes={BASEMENT_VEHICLE_TYPE_META}
        incoming={basementIn}
        outgoing={basementOut}
        allTimeIn={basementAll.incoming?.car || 0}
        allTimeOut={basementAll.outgoing?.car || 0}
        range={range}
        setRange={setRange}
        customStart={customStart}
        setCustomStart={setCustomStart}
        customEnd={customEnd}
        setCustomEnd={setCustomEnd}
        loading={loading}
        onOpenHistory={() => onOpenHistory('basement')}
      />
    </>
  );
}

/* ── Other Cameras — Live Only (Never Saved to DB) ─────────── */

function OtherCamerasLiveCard({ remaining = [] }) {
  return (
    <div className="card">
      <div className="head" style={{ flexWrap: 'wrap', gap: 8 }}>
        <div>
          <h3>📡 Other Cameras (Live Only)</h3>
          <small>Checkpoint &amp; internal live detections (in memory only)</small>
        </div>
      </div>
      <div className="body" style={{ padding: '12px 16px' }}>
        {remaining.length > 0 ? (
          <div className="vc-live-grid">
            {remaining.map(cam => {
              const inf = cam.in_frame || { car: 0, bike: 0, auto: 0, ambulance: 0, total: 0 };
              return (
                <div key={cam.stream} className="vc-live-row">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                      <span
                        style={{
                          width: 8, height: 8, borderRadius: '50%', flex: 'none',
                          background: cam.online ? '#10b981' : '#94a3b8',
                          boxShadow: cam.online ? '0 0 0 3px rgba(16,185,129,0.2)' : 'none'
                        }}
                      />
                      <b style={{ fontSize: 12, color: '#1e293b', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={cam.name}>
                        {cam.name}
                      </b>
                      <span className={`badge ${cam.role}`} style={{ fontSize: 8 }}>{formatRoleLabel(cam.role)}</span>
                      {COUNTABLE_CAMERA_ROLES.has(cam.role) && cam.count_enabled === false && (
                        <span className="badge" style={{ background: '#fef3c7', color: '#92400e', fontSize: 8 }}>DB Off</span>
                      )}
                    </div>
                    <span style={{ fontSize: 10, fontWeight: 700, color: cam.online ? '#059669' : '#94a3b8', flex: 'none' }}>
                      {cam.online ? `${inf.total} in frame` : 'Offline'}
                    </span>
                  </div>

                  <div className="vc-mini-grid">
                    {VEHICLE_TYPE_META.map(vt => (
                      <div key={vt.key} className="vc-mini-chip" title={`In view: ${inf[vt.key] || 0}`}>
                        <span>{vt.icon} {vt.label}</span>
                        <b>{inf[vt.key] || 0}</b>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty" style={{ padding: 12 }}>
            All registered Entry &amp; Exit cameras are saving directly to Gate History.
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Historical Incoming & Outgoing Security Modal ─────────── */

function exportGateHistoryToExcel({ data, scope = 'main', rangeLabel, granularity, dirFilter, typeFilter }) {
  const wb = XLSX.utils.book_new();
  const isBasement = scope === 'basement';
  const emptyBucket = { car: 0, bike: 0, auto: 0, ambulance: 0, total: 0 };
  const inc = isBasement ? (data?.basement?.incoming || emptyBucket) : (data?.incoming || emptyBucket);
  const out = isBasement ? (data?.basement?.outgoing || emptyBucket) : (data?.outgoing || emptyBucket);
  const allowedDirs = isBasement ? ['basement_incoming', 'basement_outgoing'] : ['incoming', 'outgoing'];
  const inDirKey = isBasement ? 'basement_incoming' : 'incoming';
  const outDirKey = isBasement ? 'basement_outgoing' : 'outgoing';
  const inColLabel = isBasement ? 'Basement Entry' : 'Incoming';
  const outColLabel = isBasement ? 'Basement Exit' : 'Outgoing';
  const inBucketPrefix = isBasement ? 'Entry' : 'In';
  const outBucketPrefix = isBasement ? 'Exit' : 'Out';
  const resolvedGranularity = data?.window?.granularity || granularity || 'hourly';
  const isDaily = resolvedGranularity === 'daily';

  const baseTypes = isBasement ? BASEMENT_VEHICLE_TYPE_META : VEHICLE_TYPE_META;
  const activeTypeFilter = isBasement ? 'all' : typeFilter;
  const typeEntries = activeTypeFilter === 'all'
    ? baseTypes
    : baseTypes.filter(vt => vt.key === activeTypeFilter);

  const showIn = dirFilter === 'all' || dirFilter === inDirKey;
  const showOut = dirFilter === 'all' || dirFilter === outDirKey;

  // Build Summary & Gate-wise tables
  const summaryHeader = ['Vehicle Type'];
  if (showIn) summaryHeader.push(inColLabel);
  if (showOut) summaryHeader.push(outColLabel);
  if (showIn && showOut) summaryHeader.push('Net Flow');

  const summaryRows = typeEntries.map(vt => {
    const iVal = Number(inc[vt.key] || 0);
    const oVal = Number(out[vt.key] || 0);
    const row = [vt.label];
    if (showIn) row.push(iVal);
    if (showOut) row.push(oVal);
    if (showIn && showOut) row.push(Math.max(0, iVal - oVal));
    return row;
  });
  if (activeTypeFilter === 'all') {
    const totIn = isBasement ? Number(inc.car || 0) : Number(inc.total || 0);
    const totOut = isBasement ? Number(out.car || 0) : Number(out.total || 0);
    const totRow = [isBasement ? 'Total Basement Flow' : 'Total Gate Flow'];
    if (showIn) totRow.push(totIn);
    if (showOut) totRow.push(totOut);
    if (showIn && showOut) totRow.push(Math.max(0, totIn - totOut));
    summaryRows.push(totRow);
  }

  const camFiltered = (data?.by_camera || []).filter(
    c => allowedDirs.includes(c.direction) && (dirFilter === 'all' || c.direction === dirFilter)
  );
  const camHeader = ['Gate Camera', 'Direction', ...typeEntries.map(vt => vt.label), 'Total'];
  const camRows = camFiltered.map(c => {
    const vals = typeEntries.map(vt => Number(c[vt.key] || 0));
    const total = activeTypeFilter === 'all' ? (isBasement ? Number(c.car || 0) : Number(c.total || 0)) : vals[0];
    return [c.name, getDirectionMeta(c.direction).label.toUpperCase(), ...vals, total];
  });

  // Build Time-Bucket Breakdown table (selected Hourly or Daily granularity)
  const timeHeader = ['Time Period (IST)'];
  if (showIn) {
    typeEntries.forEach(vt => timeHeader.push(`${inBucketPrefix} ${vt.label}`));
    if (activeTypeFilter === 'all') timeHeader.push(`${inBucketPrefix} Total`);
  }
  if (showOut) {
    typeEntries.forEach(vt => timeHeader.push(`${outBucketPrefix} ${vt.label}`));
    if (activeTypeFilter === 'all') timeHeader.push(`${outBucketPrefix} Total`);
  }
  if (showIn && showOut) timeHeader.push('Net Flow');

  const rawHist = isBasement ? (data?.basement?.history || []) : (data?.history || []);
  const histList = [...rawHist].reverse();
  const totalsAcc = new Array(Math.max(0, timeHeader.length - 1)).fill(0);
  const timeRows = histList.map(b => {
    const d = new Date(b.bucket);
    const lbl = isDaily
      ? d.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' })
      : d.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: true });
    const i = b.incoming || {};
    const o = b.outgoing || {};
    const nums = [];
    let inSum = 0;
    let outSum = 0;
    if (showIn) {
      typeEntries.forEach(vt => nums.push(Number(i[vt.key] || 0)));
      inSum = isBasement ? Number(i.car || 0) : (activeTypeFilter === 'all' ? Number(i.total || 0) : Number(i[activeTypeFilter] || 0));
      if (activeTypeFilter === 'all') nums.push(inSum);
    }
    if (showOut) {
      typeEntries.forEach(vt => nums.push(Number(o[vt.key] || 0)));
      outSum = isBasement ? Number(o.car || 0) : (activeTypeFilter === 'all' ? Number(o.total || 0) : Number(o[activeTypeFilter] || 0));
      if (activeTypeFilter === 'all') nums.push(outSum);
    }
    if (showIn && showOut) nums.push(Math.max(0, inSum - outSum));
    nums.forEach((n, idx) => { totalsAcc[idx] += n; });
    return [lbl, ...nums];
  });
  if (timeRows.length > 0) {
    timeRows.push(['TOTAL', ...totalsAcc]);
  }

  const timeSectionTitle = isBasement
    ? (isDaily ? 'DAILY BASEMENT ENTRY & EXIT BREAKDOWN' : 'HOURLY BASEMENT ENTRY & EXIT BREAKDOWN')
    : (isDaily ? 'DAILY INCOMING & OUTGOING BREAKDOWN' : 'HOURLY INCOMING & OUTGOING BREAKDOWN');

  const dirFilterDisplay = dirFilter === 'all'
    ? (isBasement ? 'All Basement Directions' : 'All Main Gate Directions')
    : getDirectionMeta(dirFilter).label;

  // Sheet 1: Complete Gate Audit Overview (Summary + Gate-wise + Selected Hourly/Daily Breakdown)
  const nowIst = new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', hour12: true });
  const sheet1AoA = [
    [isBasement ? 'Medanta Hospital — Basement Gate Vehicle Counts & Audit Log' : 'Medanta Hospital — Gate Vehicle Counts & Security Audit'],
    ['Generated (IST)', nowIst, 'Window', rangeLabel],
    ['Granularity', isDaily ? 'Daily Buckets' : 'Hourly Buckets', 'Direction', dirFilterDisplay],
    ['Vehicle Type Filter', isBasement ? 'Cars Only (Basement)' : (typeFilter === 'all' ? 'All Vehicle Types' : typeFilter.toUpperCase())],
    [],
    ['SUMMARY BY VEHICLE TYPE'],
    summaryHeader,
    ...summaryRows,
    [],
    ['GATE-WISE BREAKDOWN'],
    camHeader,
    ...(camRows.length > 0 ? camRows : [['No gate breakdown records in selected window']]),
    [],
    [timeSectionTitle],
    timeHeader,
    ...(timeRows.length > 0 ? timeRows : [['No historical gate vehicle records in selected window']])
  ];
  const maxColsS1 = Math.max(camHeader.length, timeHeader.length, 7);
  const ws1 = XLSX.utils.aoa_to_sheet(sheet1AoA);
  ws1['!cols'] = [{ wch: 28 }, { wch: 20 }, ...new Array(maxColsS1 - 2).fill(0).map(() => ({ wch: 14 }))];
  XLSX.utils.book_append_sheet(wb, ws1, isBasement ? 'Basement Audit Summary' : 'Gate Audit Summary');

  // Sheet 2: Dedicated Selected Hourly or Daily Breakdown Tab
  const ws2 = XLSX.utils.aoa_to_sheet([timeHeader, ...(timeRows.length > 0 ? timeRows : [['No records']])]);
  ws2['!cols'] = [{ wch: 26 }, ...timeHeader.slice(1).map(() => ({ wch: 14 }))];
  XLSX.utils.book_append_sheet(wb, ws2, isDaily ? 'Daily Breakdown' : 'Hourly Breakdown');

  // Sheet 3: Tracked Gate Events Log
  const evFiltered = (data?.recent_events || []).filter(ev => {
    if (!allowedDirs.includes(ev.direction)) return false;
    if (dirFilter !== 'all' && ev.direction !== dirFilter) return false;
    if (isBasement && ev.vehicle_type !== 'car') return false;
    if (!isBasement && typeFilter !== 'all' && ev.vehicle_type !== typeFilter) return false;
    return true;
  });
  const evHeaders = ['S.No', 'Date & Time (IST)', 'Direction', 'Gate Camera', 'Gate Group', 'Matched Cameras', 'Vehicle Type', 'License Plate', 'Confidence (%)'];
  const evRows = evFiltered.map((ev, idx) => [
    idx + 1,
    tsDate(ev.ts),
    getDirectionMeta(ev.direction).label.toUpperCase(),
    ev.camera || '—',
    ev.gate_group || 'Default',
    (ev.merged_cameras && ev.merged_cameras.length > 0) ? ev.merged_cameras.join(', ') : (ev.camera || '—'),
    (ev.vehicle_type || '').toUpperCase(),
    ev.plate || '—',
    ev.confidence != null ? Math.round(ev.confidence * 100) : '—'
  ]);
  const ws3 = XLSX.utils.aoa_to_sheet([evHeaders, ...evRows]);
  ws3['!cols'] = [
    { wch: 7 }, { wch: 24 }, { wch: 16 }, { wch: 24 },
    { wch: 16 }, { wch: 28 }, { wch: 14 }, { wch: 16 }, { wch: 14 }
  ];
  XLSX.utils.book_append_sheet(wb, ws3, isBasement ? 'Basement Events' : 'Tracked Gate Events');

  const now = new Date();
  const istDate = now.toLocaleDateString('en-CA', { timeZone: 'Asia/Kolkata' });
  const istTime = now.toLocaleTimeString('en-GB', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hour12: false }).replace(':', '-');
  const filePrefix = isBasement ? 'Medanta_Basement_Gate_Counts_' : 'Medanta_Gate_Vehicle_Counts_';
  XLSX.writeFile(wb, `${filePrefix}${istDate}_${istTime}.xlsx`);
  return evFiltered.length;
}

function VehicleHistoryModal({ open, scope = 'main', onClose, initialData }) {
  const isBasement = scope === 'basement';
  const allowedDirs = isBasement ? ['basement_incoming', 'basement_outgoing'] : ['incoming', 'outgoing'];
  const [histRange, setHistRange] = useState('24h');
  const [fromStr, setFromStr] = useState(() => {
    const d = new Date(Date.now() - 7 * 86400000);
    return d.toISOString().slice(0, 16);
  });
  const [toStr, setToStr] = useState(() => new Date().toISOString().slice(0, 16));
  const [granularity, setGranularity] = useState('hourly');
  const [dirFilter, setDirFilter] = useState('all');
  const [typeFilter, setTypeFilter] = useState('all');
  const [histData, setHistData] = useState(initialData || null);
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportMsg, setExportMsg] = useState(null);

  useEffect(() => {
    if (!open) return;
    setDirFilter('all');
    setTypeFilter('all');
    setHistData(initialData || null);
  }, [open, scope]);

  const buildQuery = useCallback((limitVal = 200, includeTypeFilter = false) => {
    let qs = `granularity=${granularity}&limit=${limitVal}`;
    if (histRange === 'custom') {
      const fMs = new Date(fromStr).getTime() || 0;
      const tMs = new Date(toStr).getTime() || Date.now();
      qs += `&from_ts=${fMs}&to_ts=${tMs}`;
    } else {
      qs += `&hours=${histRange === '7d' ? '168' : histRange === '30d' ? '720' : histRange === '24h' ? '24' : histRange}`;
    }
    const effectiveDir = dirFilter !== 'all' ? dirFilter : scope;
    qs += `&direction=${encodeURIComponent(effectiveDir)}`;
    if (isBasement) {
      qs += '&type=car';
    } else if (includeTypeFilter && typeFilter !== 'all') {
      qs += `&type=${encodeURIComponent(typeFilter)}`;
    }
    return qs;
  }, [scope, isBasement, histRange, fromStr, toStr, granularity, dirFilter, typeFilter]);

  const histReqRef = useRef(0);
  const fetchHistory = useCallback(async () => {
    if (!open) return;
    const reqId = ++histReqRef.current;
    setLoading(true);
    try {
      const res = await api('/vehicle-counts?' + buildQuery(200, false));
      if (reqId === histReqRef.current) setHistData(res);
    } catch (e) {
      console.error('Failed to load historical vehicle counts:', e);
    } finally {
      if (reqId === histReqRef.current) setLoading(false);
    }
  }, [open, buildQuery]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  if (!open) return null;

  const emptyBucket = { car: 0, bike: 0, auto: 0, ambulance: 0, total: 0 };
  const modalTypes = isBasement ? BASEMENT_VEHICLE_TYPE_META : VEHICLE_TYPE_META;
  const incoming = isBasement ? (histData?.basement?.incoming || emptyBucket) : (histData?.incoming || emptyBucket);
  const outgoing = isBasement ? (histData?.basement?.outgoing || emptyBucket) : (histData?.outgoing || emptyBucket);
  const byCamera = (histData?.by_camera || []).filter(
    c => allowedDirs.includes(c.direction) && (dirFilter === 'all' || c.direction === dirFilter)
  );
  const rawHist = isBasement ? (histData?.basement?.history || []) : (histData?.history || []);
  const historyRows = [...rawHist].reverse();
  const events = (histData?.recent_events || []).filter(ev => {
    if (!allowedDirs.includes(ev.direction)) return false;
    if (dirFilter !== 'all' && ev.direction !== dirFilter) return false;
    if (isBasement && ev.vehicle_type !== 'car') return false;
    if (!isBasement && typeFilter !== 'all' && ev.vehicle_type !== typeFilter) return false;
    return true;
  });
  const inModalTotal = isBasement ? (incoming.car || 0) : (incoming.total || 0);
  const outModalTotal = isBasement ? (outgoing.car || 0) : (outgoing.total || 0);
  const hasAnyRecords = inModalTotal > 0 || outModalTotal > 0 || historyRows.length > 0 || events.length > 0;

  const handleDownloadExcel = async () => {
    if (!hasAnyRecords || exporting) return;
    setExporting(true);
    try {
      let fullData = histData;
      try {
        fullData = await api('/vehicle-counts?' + buildQuery(10000, true));
      } catch {
        // Fallback to already loaded modal data if full fetch fails
      }
      const rangeNames = {
        today: 'Today (IST)',
        '24h': 'Last 24 Hours',
        '7d': 'Last 7 Days',
        '30d': 'Last 30 Days',
        all: 'All Time',
        custom: `${fromStr.replace('T', ' ')} to ${toStr.replace('T', ' ')}`
      };
      const evCount = exportGateHistoryToExcel({
        data: fullData,
        scope,
        rangeLabel: rangeNames[histRange] || histRange,
        granularity,
        dirFilter,
        typeFilter
      });
      setExportMsg(`✅ Exported Excel (${evCount} events)`);
      setTimeout(() => setExportMsg(null), 4000);
    } catch (err) {
      alert('Error generating Excel: ' + err.message);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" style={{ width: 920, maxWidth: '95vw', maxHeight: '90vh', display: 'flex', flexDirection: 'column' }} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '15px 20px', borderBottom: '1px solid #eef2f7', background: '#fbfcfe', gap: 10, flexWrap: 'wrap' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 16, fontWeight: 800, color: '#172138' }}>
              {isBasement ? '🅿️ Historical Basement Gate Counts & Audit Log' : '🛡️ Historical Gate Vehicle Counts & Security Audit Log'}
            </h3>
            <small style={{ color: '#8ca0bc', fontSize: 11 }}>
              {isBasement
                ? 'Basement Entry & Basement Exit breakdown saved permanently in database'
                : 'Incoming & outgoing vehicle breakdown saved permanently in database'}
            </small>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            {exportMsg && (
              <span className="badge" style={{ background: '#e3f6ef', color: '#17835c', fontSize: 11, fontWeight: 700 }}>
                {exportMsg}
              </span>
            )}
            <button
              type="button"
              className="btn sm"
              onClick={handleDownloadExcel}
              disabled={exporting || !hasAnyRecords}
              style={{
                background: hasAnyRecords ? '#17835c' : '#cbd5e1',
                color: '#fff',
                borderColor: hasAnyRecords ? '#17835c' : '#cbd5e1',
                fontWeight: 700,
                fontSize: 11,
                padding: '6px 14px',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                cursor: (hasAnyRecords && !exporting) ? 'pointer' : 'not-allowed',
                boxShadow: hasAnyRecords ? '0 2px 8px -2px rgba(23,131,92,0.45)' : 'none'
              }}
              title="Download Gate Summary, Time Breakdown & Tracked Events as an Excel (.xlsx) workbook"
            >
              {exporting ? <span className="spin-ring" /> : <span>📥</span>}
              <span>{exporting ? 'Generating Excel…' : 'Download Excel'}</span>
            </button>
            <button className="btn sm" onClick={onClose} style={{ borderRadius: '50%', width: 30, height: 30, padding: 0 }}>✕</button>
          </div>
        </div>

        {/* Filter Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 20px', background: '#f8fafd', borderBottom: '1px solid #eef2f7', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {[
              { k: 'today', label: 'Today' },
              { k: '24h', label: '24h' },
              { k: '7d', label: '7 Days' },
              { k: '30d', label: '30 Days' },
              { k: 'all', label: 'All Time' },
              { k: 'custom', label: '📅 Custom' },
            ].map(b => (
              <button key={b.k} type="button" className={`vc-range-btn ${histRange === b.k ? 'active' : ''}`} onClick={() => setHistRange(b.k)}>
                {b.label}
              </button>
            ))}
          </div>
          <select value={granularity} onChange={e => setGranularity(e.target.value)} style={{ padding: '4px 8px', border: '1px solid #dfe6f0', borderRadius: 6, fontSize: 11, fontWeight: 700 }}>
            <option value="hourly">Hourly Buckets</option>
            <option value="daily">Daily Buckets</option>
          </select>
          <select value={dirFilter} onChange={e => setDirFilter(e.target.value)} style={{ padding: '4px 8px', border: '1px solid #dfe6f0', borderRadius: 6, fontSize: 11, fontWeight: 700 }}>
            {isBasement ? (
              <>
                <option value="all">All Basement Directions</option>
                <option value="basement_incoming">🟠 Basement Entry</option>
                <option value="basement_outgoing">🟡 Basement Exit</option>
              </>
            ) : (
              <>
                <option value="all">All Main Gate Directions</option>
                <option value="incoming">🟢 Main Gate Incoming</option>
                <option value="outgoing">🔵 Main Gate Outgoing</option>
              </>
            )}
          </select>
          {!isBasement && (
            <select value={typeFilter} onChange={e => setTypeFilter(e.target.value)} style={{ padding: '4px 8px', border: '1px solid #dfe6f0', borderRadius: 6, fontSize: 11, fontWeight: 700 }}>
              <option value="all">All Vehicle Types</option>
              <option value="car">🚗 Cars</option>
              <option value="bike">🏍️ Bikes</option>
              <option value="auto">🛺 Autos</option>
              <option value="ambulance">🚑 Ambulances</option>
            </select>
          )}
          <LoadingPill show={loading} text="Loading…" />
        </div>

        {histRange === 'custom' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 20px', background: '#f0f9f5', borderBottom: '1px solid #cce8de', flexWrap: 'wrap' }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: '#17835c' }}>Custom Window:</span>
            <input type="datetime-local" value={fromStr} onChange={e => setFromStr(e.target.value)} style={{ border: '1px solid #bfe2d4', borderRadius: 6, fontSize: 11, padding: '3px 6px' }} />
            <span style={{ fontSize: 11, color: '#64748b' }}>to</span>
            <input type="datetime-local" value={toStr} onChange={e => setToStr(e.target.value)} style={{ border: '1px solid #bfe2d4', borderRadius: 6, fontSize: 11, padding: '3px 6px' }} />
          </div>
        )}

        <div className="loading-wrap" style={{ position: 'relative', flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
          <LoadingOverlay show={loading} text="Loading historical records…" />
          <div style={{ padding: 20, overflowY: 'auto', display: 'grid', gap: 18 }}>
          {/* Summary Totals Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10 }}>
            {modalTypes.map(vt => (
              <div key={vt.key} style={{ background: vt.bg, border: '1px solid #e2e8f0', borderRadius: 10, padding: '10px 14px' }}>
                <div style={{ fontSize: 11, fontWeight: 800, color: vt.color }}>{vt.icon} {vt.label.toUpperCase()}</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 12 }}>
                  <span>{isBasement ? 'Entry:' : 'In:'} <b style={{ color: isBasement ? '#c2410c' : '#15803d' }}>{incoming[vt.key] || 0}</b></span>
                  <span>{isBasement ? 'Exit:' : 'Out:'} <b style={{ color: isBasement ? '#a16207' : '#1d4ed8' }}>{outgoing[vt.key] || 0}</b></span>
                </div>
              </div>
            ))}
          </div>

          {/* Per-Gate Breakdown */}
          {byCamera.length > 0 && (
            <div>
              <h4 style={{ margin: '0 0 8px', fontSize: 12, color: '#334155', textTransform: 'uppercase', letterSpacing: .4 }}>Gate-wise Breakdown</h4>
              <table className="vc-matrix" style={{ border: '1px solid #eef2f7', borderRadius: 8 }}>
                <thead style={{ background: '#f8fafd' }}>
                  <tr>
                    <th>Gate Camera</th>
                    <th>Direction</th>
                    {modalTypes.map(vt => (
                      <th key={vt.key} className="num">{vt.icon} {vt.singular}</th>
                    ))}
                    <th className="num">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {byCamera.map((c, idx) => {
                    const dm = getDirectionMeta(c.direction);
                    const camTotal = isBasement ? (c.car || 0) : (c.total || 0);
                    return (
                      <tr key={c.stream + c.direction + idx}>
                        <td><b>{c.name}</b></td>
                        <td><span className={`vc-pill ${dm.pill}`}>{dm.label}</span></td>
                        {modalTypes.map(vt => (
                          <td key={vt.key} className="num">{c[vt.key] || 0}</td>
                        ))}
                        <td className="num"><b>{camTotal}</b></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Time-Bucket History Table */}
          <div>
            <h4 style={{ margin: '0 0 8px', fontSize: 12, color: '#334155', textTransform: 'uppercase', letterSpacing: .4 }}>
              {histData?.window?.granularity === 'daily' ? 'Daily' : 'Hourly'} {isBasement ? 'Basement Entry & Exit' : 'Incoming & Outgoing'} Breakdown
            </h4>
            {historyRows.length > 0 ? (
              <div style={{ maxHeight: 210, overflowY: 'auto', border: '1px solid #eef2f7', borderRadius: 8 }}>
                <table className="vc-matrix" style={{ marginTop: 0 }}>
                  <thead style={{ position: 'sticky', top: 0, background: '#f8fafd' }}>
                    <tr>
                      <th>Time Period (IST)</th>
                      {!isBasement && <th className="num">In (Car / Bike / Auto / Amb)</th>}
                      <th className="num">{isBasement ? '🟠 Basement Entry (Cars)' : '🟢 In Total'}</th>
                      {!isBasement && <th className="num">Out (Car / Bike / Auto / Amb)</th>}
                      <th className="num">{isBasement ? '🟡 Basement Exit (Cars)' : '🔵 Out Total'}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {historyRows.map(b => {
                      const d = new Date(b.bucket);
                      const lbl = histData?.window?.granularity === 'daily'
                        ? d.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric' })
                        : d.toLocaleString('en-IN', { timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit', hour12: true });
                      const i = b.incoming, o = b.outgoing;
                      const bInTotal = isBasement ? (i.car || 0) : (i.total || 0);
                      const bOutTotal = isBasement ? (o.car || 0) : (o.total || 0);
                      return (
                        <tr key={b.bucket}>
                          <td>{lbl}</td>
                          {!isBasement && <td className="num">{i.car} / {i.bike} / {i.auto} / {i.ambulance}</td>}
                          <td className="num"><span className={`vc-pill ${isBasement ? 'base-in' : 'in'}`}>{bInTotal}</span></td>
                          {!isBasement && <td className="num">{o.car} / {o.bike} / {o.auto} / {o.ambulance}</td>}
                          <td className="num"><span className={`vc-pill ${isBasement ? 'base-out' : 'out'}`}>{bOutTotal}</span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty" style={{ border: '1px solid #eef2f7', borderRadius: 8 }}>
                {isBasement ? 'No historical basement gate records in this selected window yet.' : 'No historical gate vehicle records in this selected window yet.'}
              </div>
            )}
          </div>

          {/* Detailed Security Audit Events */}
          <div>
            <h4 style={{ margin: '0 0 8px', fontSize: 12, color: '#334155', textTransform: 'uppercase', letterSpacing: .4 }}>
              Recent Tracked Gate Events ({events.length})
            </h4>
            {events.length > 0 ? (
              <div style={{ maxHeight: 220, overflowY: 'auto', border: '1px solid #eef2f7', borderRadius: 8 }}>
                <table className="vc-matrix" style={{ marginTop: 0 }}>
                  <thead style={{ position: 'sticky', top: 0, background: '#f8fafd' }}>
                    <tr>
                      <th>Time (IST)</th>
                      <th>Direction</th>
                      <th>Gate Camera</th>
                      <th>Type</th>
                      <th>Plate (If Read)</th>
                      <th className="num">Conf</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.map(ev => {
                      const extraCams = (ev.merged_cameras && ev.merged_cameras.length > 1) ? ev.merged_cameras.length - 1 : 0;
                      const dm = getDirectionMeta(ev.direction);
                      return (
                        <tr key={ev.id}>
                          <td style={{ whiteSpace: 'nowrap' }}>{tsDate(ev.ts)}</td>
                          <td><span className={`vc-pill ${dm.pill}`}>{dm.label}</span></td>
                          <td>
                            <span>{ev.camera}</span>
                            {extraCams > 0 && (
                              <span
                                className="badge"
                                style={{ marginLeft: 5, background: '#dcfce7', color: '#166534', fontSize: 8 }}
                                title={`Merged across ${ev.merged_cameras.join(', ')}`}
                              >
                                +{extraCams} cam
                              </span>
                            )}
                            {ev.gate_group && (
                              <span style={{ marginLeft: 5, fontSize: 9, color: '#64748b' }}>({ev.gate_group})</span>
                            )}
                          </td>
                          <td style={{ textTransform: 'capitalize' }}>{ev.vehicle_type}</td>
                          <td><b>{ev.plate || '—'}</b></td>
                          <td className="num">{ev.confidence != null ? `${Math.round(ev.confidence * 100)}%` : '—'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty" style={{ border: '1px solid #eef2f7', borderRadius: 8 }}>No tracked events matching filter.</div>
            )}
          </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Ambulance Email Dispatch Modal (matches 31030/31040 flow) ── */

function AmbulanceEmailDispatchModal({ alertData, onClose, onDispatched }) {
  const [toInput, setToInput] = useState('');
  const [subject, setSubject] = useState('');
  const [note, setNote] = useState('');
  const [sending, setSending] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const quickPresets = [
    { label: '+ Emergency Desk', email: 'emergency.desk@medanta.org' },
    { label: '+ Security Control', email: 'security.desk@medanta.org' },
    { label: '+ Ambulance Desk', email: 'ambulance.dispatch@medanta.org' },
    { label: '+ Operations Lead', email: 'ops.lead@medanta.org' },
    { label: '+ My Email', email: 'bellwetherinbox01@gmail.com' },
  ];

  useEffect(() => {
    if (!alertData) return;
    setSubject(alertData.subject || '');
    setNote('');
    setSending(false);
    setFeedback(null);
  }, [alertData]);

  useEffect(() => {
    if (!alertData) return;
    const onKey = (e) => { if (e.key === 'Escape' && !sending) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [alertData, sending, onClose]);

  if (!alertData) return null;

  const addPresetEmail = (email) => {
    const existing = toInput.split(',').map(x => x.trim()).filter(Boolean);
    if (!existing.includes(email)) {
      existing.push(email);
      setToInput(existing.join(', '));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const recipients = toInput.split(',').map(x => x.trim()).filter(x => x && x.includes('@'));
    if (recipients.length === 0) {
      setFeedback({ type: 'error', message: 'Please enter at least one valid recipient email address.' });
      return;
    }
    setSending(true);
    setFeedback(null);
    try {
      const res = await api('/ambulances/dispatch-email', {
        ref_id: alertData.ref_id || '',
        plate: alertData.plate || '',
        to: recipients,
        subject: subject.trim(),
        title: alertData.title,
        summary: alertData.summary,
        status: alertData.status,
        metadata: alertData.metadata || {},
        note: note.trim(),
        entry_sighting_id: alertData.entry_sighting_id || null,
        exit_sighting_id: alertData.exit_sighting_id || null,
      });
      if (!res || res.error || !res.success) {
        throw new Error(res?.error || res?.detail || 'Failed to dispatch ambulance alert email.');
      }
      setFeedback({
        type: 'success',
        message: `Ambulance dispatch email sent to ${recipients.length} recipient(s)!`,
        messageId: res.message_id || ''
      });
      if (onDispatched) onDispatched(res);
    } catch (err) {
      setFeedback({
        type: 'error',
        message: err.message || 'Network error while reaching Email Dispatch service.'
      });
    } finally {
      setSending(false);
    }
  };

  const hasPlateImg = Boolean(alertData.plate_url);
  const hasPhotoImg = Boolean(alertData.photo_url);

  return (
    <div
      style={{
        position: 'fixed', inset: 0, backgroundColor: 'rgba(15, 23, 42, 0.75)',
        backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center',
        justifyContent: 'center', zIndex: 99999, padding: 16
      }}
      onClick={() => !sending && onClose()}
    >
      <div
        style={{
          backgroundColor: '#ffffff', borderRadius: 14, width: '100%', maxWidth: 680,
          maxHeight: '90vh', overflowY: 'auto', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)',
          border: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column'
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{
          padding: '16px 20px', borderBottom: '1px solid #f1f5f9', display: 'flex',
          alignItems: 'center', justifyContent: 'space-between', backgroundColor: '#f8fafc',
          borderTopLeftRadius: 14, borderTopRightRadius: 14
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 20 }}>✉️</span>
            <div>
              <div style={{ fontSize: 15, fontWeight: 800, color: '#0f172a' }}>
                Dispatch Ambulance Alert Email
              </div>
              <div style={{ fontSize: 11, color: '#64748b' }}>
                Sends formal ambulance movement advisory via AWS SES (Port 31040) with vehicle &amp; plate capture
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={sending}
            style={{ border: 'none', background: 'transparent', fontSize: 18, cursor: 'pointer', color: '#94a3b8', padding: '4px 8px', borderRadius: 6 }}
          >
            ✕
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
          {feedback && (
            <div style={{
              padding: '12px 16px', borderRadius: 8,
              backgroundColor: feedback.type === 'success' ? '#f0fdf4' : '#fef2f2',
              border: `1px solid ${feedback.type === 'success' ? '#bbf7d0' : '#fecaca'}`,
              color: feedback.type === 'success' ? '#166534' : '#991b1b',
              fontSize: 13, display: 'flex', flexDirection: 'column', gap: 4
            }}>
              <div style={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>{feedback.type === 'success' ? '✓' : '⚠️'}</span>
                <span>{feedback.message}</span>
              </div>
              {feedback.messageId && (
                <div style={{ fontSize: 11, fontFamily: 'monospace', color: '#15803d' }}>
                  SES Message ID: {feedback.messageId}
                </div>
              )}
            </div>
          )}

          {/* Recipients */}
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#334155', marginBottom: 6 }}>
              Send To (Email Address or comma-separated list) *
            </label>
            <input
              type="text"
              required
              placeholder="e.g. emergency.desk@medanta.org, security.desk@medanta.org"
              value={toInput}
              onChange={e => setToInput(e.target.value)}
              disabled={sending}
              style={{ width: '100%', padding: '9px 12px', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 13, outline: 'none', boxSizing: 'border-box' }}
            />
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginTop: 6 }}>
              <span style={{ fontSize: 11, color: '#64748b', fontWeight: 600 }}>Quick add:</span>
              {quickPresets.map(p => (
                <button
                  key={p.email}
                  type="button"
                  onClick={() => addPresetEmail(p.email)}
                  disabled={sending}
                  style={{
                    border: '1px solid #e2e8f0', backgroundColor: '#f8fafc', color: '#475569',
                    fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 9999, cursor: 'pointer'
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Subject */}
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#334155', marginBottom: 6 }}>
              Subject Line *
            </label>
            <input
              type="text"
              required
              value={subject}
              onChange={e => setSubject(e.target.value)}
              disabled={sending}
              style={{ width: '100%', padding: '9px 12px', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 13, outline: 'none', boxSizing: 'border-box' }}
            />
          </div>

          {/* Attached Visual Evidence */}
          {(hasPlateImg || hasPhotoImg) && (
            <div style={{ backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 10, padding: '12px 14px' }}>
              <div style={{ fontSize: 11, fontWeight: 800, color: '#475569', textTransform: 'uppercase', marginBottom: 8, letterSpacing: 0.4 }}>
                📷 Attached Visual Evidence
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: (hasPlateImg && hasPhotoImg) ? '1fr 1fr' : '1fr', gap: 12 }}>
                {hasPlateImg && (
                  <div style={{ textAlign: 'center', backgroundColor: '#fff', padding: 8, borderRadius: 8, border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>🪪 LICENSE PLATE CROP</div>
                    <img
                      src={`${API}${alertData.plate_url}`}
                      alt="License Plate Crop"
                      style={{ height: 100, maxWidth: '100%', objectFit: 'contain', borderRadius: 6, border: '1px solid #cbd5e1', display: 'inline-block' }}
                    />
                  </div>
                )}
                {hasPhotoImg && (
                  <div style={{ textAlign: 'center', backgroundColor: '#fff', padding: 8, borderRadius: 8, border: '1px solid #e2e8f0' }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>🚑 AMBULANCE SNAPSHOT</div>
                    <img
                      src={`${API}${alertData.photo_url}`}
                      alt="Ambulance Vehicle"
                      style={{ height: 110, maxWidth: '100%', objectFit: 'contain', borderRadius: 6, border: '1px solid #cbd5e1', display: 'inline-block' }}
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Metadata Summary */}
          {alertData.metadata && Object.keys(alertData.metadata).length > 0 && (
            <div style={{ backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 14px' }}>
              <div style={{ fontSize: 11, fontWeight: 800, color: '#475569', textTransform: 'uppercase', marginBottom: 6 }}>
                📋 Ambulance Movement Metadata
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 8 }}>
                {Object.entries(alertData.metadata).map(([k, v]) => (
                  <div key={k} style={{ fontSize: 11 }}>
                    <span style={{ color: '#64748b', fontWeight: 600 }}>{k}: </span>
                    <span style={{ color: '#0f172a', fontWeight: 700 }}>{String(v)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Dispatcher Note */}
          <div>
            <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#334155', marginBottom: 6 }}>
              Dispatcher Note / Instructions (Optional)
            </label>
            <textarea
              rows={2}
              placeholder="e.g. Ambulance arrived at Main Gate; please clear Emergency Bay & notify ER triage desk."
              value={note}
              onChange={e => setNote(e.target.value)}
              disabled={sending}
              style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #cbd5e1', fontSize: 12, outline: 'none', boxSizing: 'border-box', resize: 'none' }}
            />
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 10, borderTop: '1px solid #f1f5f9', paddingTop: 16, marginTop: 4 }}>
            <button
              type="button"
              onClick={onClose}
              disabled={sending}
              style={{ padding: '8px 16px', borderRadius: 8, border: '1px solid #cbd5e1', backgroundColor: '#fff', color: '#475569', fontSize: 12, fontWeight: 700, cursor: 'pointer' }}
            >
              {feedback?.type === 'success' ? 'Close' : 'Cancel'}
            </button>
            <button
              type="submit"
              disabled={sending}
              style={{
                padding: '8px 20px', borderRadius: 8, border: 'none',
                backgroundColor: sending ? '#94a3b8' : '#0284c7', color: '#fff',
                fontSize: 12, fontWeight: 700, cursor: sending ? 'not-allowed' : 'pointer'
              }}
            >
              {sending ? 'Dispatching…' : '✉️ Dispatch Email'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

/* ── Bottom Full-Width Ambulance List & Dispatch Card ──────── */

function AmbulanceRegistryCard({ data, rangeHours, setRangeHours, onRefresh, onOpenSightingPhoto, onOpenDispatch, loading = false }) {
  const [statusFilter, setStatusFilter] = useState('all'); // 'all' | 'on_site' | 'exited'
  const [searchText, setSearchText] = useState('');

  const rows = data?.ambulances || [];
  const summary = data?.summary || { total: 0, on_site: 0, exited: 0, plated: 0, unplated: 0, dispatched: 0 };

  const filtered = useMemo(() => {
    const q = searchText.trim().toUpperCase();
    return rows.filter(a => {
      if (statusFilter !== 'all' && a.status !== statusFilter) return false;
      if (q) {
        const plate = (a.plate || 'PLATE NOT READ').toUpperCase();
        const cams = (a.cameras || []).map(c => (c.name || c.stream || '').toUpperCase()).join(' ');
        if (!plate.includes(q) && !cams.includes(q)) return false;
      }
      return true;
    });
  }, [rows, statusFilter, searchText]);

  const buildDispatchPayload = (amb) => {
    const isOnSite = amb.status === 'on_site';
    const plateDisp = amb.plate || 'Unplated Ambulance';
    const entryCam = amb.entry?.device || amb.cameras?.[0]?.name || 'Gate Camera';
    const entryTimeStr = amb.entry?.at ? tsDate(amb.entry.at) : (amb.first_at ? tsDate(amb.first_at) : '—');
    const exitTimeStr = isOnSite
      ? 'Still On Premises (Has not exited yet)'
      : (amb.exit_confirmed ? `${tsDate(amb.exit?.at || amb.last_at)} (${amb.exit?.device || 'Exit Gate'})` : `${tsDate(amb.exit?.at || amb.last_at)} (Inferred Exit — No Exit Read)`);
    const dwellStr = amb.source === 'journey' ? dwell(amb.dwell_ms) : (amb.entry ? 'Gate Entry Event' : 'Gate Exit Event');
    const confStr = amb.confidence ? `${Math.round(amb.confidence * 100)}% (${amb.reads || 1} read${(amb.reads || 1) !== 1 ? 's' : ''})` : `${amb.reads || 1} detection`;

    return {
      ref_id: amb.id,
      plate: amb.plate || '',
      status: amb.status,
      entry_sighting_id: amb.entry_sighting_id || null,
      exit_sighting_id: amb.exit_sighting_id || null,
      photo_url: amb.photo_url || null,
      plate_url: amb.plate_url || null,
      subject: `🚑 Medanta Ambulance Alert: ${plateDisp} — ${isOnSite ? 'On Premises' : 'Exited'} (${entryCam})`,
      title: `Emergency Vehicle Tracking — Ambulance ${isOnSite ? 'Arrival / On Premises' : 'Departure Advisory'}`,
      summary: isOnSite
        ? `Ambulance ${plateDisp} entered hospital premises via ${entryCam} at ${entryTimeStr} and is currently on site.`
        : `Ambulance ${plateDisp} completed gate movement and departed at ${amb.exit?.at ? tsDate(amb.exit.at) : tsDate(amb.last_at)}.`,
      metadata: {
        'Vehicle Type': 'Ambulance',
        'License Plate': amb.plate || 'Plate Not Read',
        'Current Status': isOnSite ? 'On Premises' : 'Exited',
        'Arrival / Entry': `${entryTimeStr} (${entryCam})`,
        'Exit / Last Seen': exitTimeStr,
        'Dwell Duration': dwellStr,
        'Detection Confidence': confStr
      }
    };
  };

  return (
    <div className="card" style={{ marginTop: 16 }}>
      {/* Header */}
      <div className="head" style={{ flexWrap: 'wrap', gap: 12, padding: '14px 18px', background: 'linear-gradient(90deg, #fff5f5 0%, #ffffff 60%)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10, background: '#fee2e2', border: '1px solid #fecaca',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18
          }}>
            🚑
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: 16, color: '#7f1d1d', display: 'flex', alignItems: 'center', gap: 8 }}>
              Ambulance Registry &amp; Email Dispatch
            </h3>
            <small style={{ color: '#64748b' }}>
              All detected ambulances (plated journeys &amp; gate detections) with instant AWS SES dispatch alerts
            </small>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <LoadingPill show={loading} text="Loading…" />
          <span className="badge" style={{ background: '#fee2e2', color: '#b91c1c', border: '1px solid #fca5a5', fontSize: 10, fontWeight: 800 }}>
            ● {summary.on_site} On Premises
          </span>
          <span className="badge" style={{ background: '#dcfce7', color: '#15803d', border: '1px solid #86efac', fontSize: 10, fontWeight: 800 }}>
            ✓ {summary.exited} Exited
          </span>
          <span className="badge" style={{ background: '#f1f5f9', color: '#334155', fontSize: 10, fontWeight: 700 }}>
            Total: {summary.total}
          </span>
          {summary.dispatched > 0 && (
            <span className="badge" style={{ background: '#e0f2fe', color: '#0369a1', fontSize: 10, fontWeight: 700 }}>
              ✉ {summary.dispatched} Dispatched
            </span>
          )}
        </div>
      </div>

      {/* Filter Bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
        padding: '10px 18px', background: '#fafbfc', borderBottom: '1px solid #eef2f7', flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', gap: 4 }}>
            {[
              { k: '1', label: '1h' },
              { k: 'today', label: 'Today' },
              { k: '24', label: '24h' },
              { k: '168', label: '7 Days' },
              { k: '720', label: '30 Days' },
            ].map(r => (
              <button
                key={r.k}
                type="button"
                className={`vc-range-btn ${String(rangeHours) === r.k ? 'active' : ''}`}
                onClick={() => setRangeHours(r.k)}
              >
                {r.label}
              </button>
            ))}
          </div>

          <select
            value={statusFilter}
            onChange={e => setStatusFilter(e.target.value)}
            style={{ padding: '4px 8px', border: '1px solid #dfe6f0', borderRadius: 6, fontSize: 11, fontWeight: 700, background: '#fff' }}
          >
            <option value="all">All Statuses ({rows.length})</option>
            <option value="on_site">🔴 On Premises ({summary.on_site})</option>
            <option value="exited">🟢 Exited ({summary.exited})</option>
          </select>

          <input
            placeholder="🔍 Search plate or camera…"
            value={searchText}
            onChange={e => setSearchText(e.target.value)}
            style={{ padding: '5px 10px', border: '1px solid #dfe6f0', borderRadius: 6, fontSize: 11, fontWeight: 600, width: 190, background: '#fff' }}
          />
        </div>

        <button type="button" className="btn sm" onClick={onRefresh} disabled={loading} style={{ fontSize: 11, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
          {loading ? <span className="spin-ring" /> : <span>🔄</span>}
          <span>{loading ? 'Refreshing…' : 'Refresh'}</span>
        </button>
      </div>

      {/* Table */}
      <div className="loading-wrap">
        <LoadingOverlay show={loading} text="Loading ambulances…" />
        <div style={{ overflowX: 'auto', maxHeight: 420, overflowY: 'auto' }}>
        <table className="tbl">
          <thead style={{ position: 'sticky', top: 0, background: '#fff', zIndex: 2 }}>
            <tr>
              <th>Ambulance / Plate</th>
              <th>Entry / Arrival Gate</th>
              <th>Exit / Current Status</th>
              <th>Dwell / Flow</th>
              <th>Reads &amp; Dispatch History</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(amb => {
              const isOnSite = amb.status === 'on_site';
              const plateStr = amb.plate || 'PLATE NOT READ';
              return (
                <tr key={amb.id}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div
                        className={amb.photo_url ? 'clickable-thumb' : undefined}
                        onClick={() => amb.photo_url && onOpenSightingPhoto({
                          photo_url: amb.photo_url,
                          plate: plateStr,
                          stream: amb.entry?.stream || amb.exit?.stream || 'ambulance',
                          ts: amb.last_at || amb.first_at,
                          confidence: amb.confidence
                        })}
                        style={{
                          width: 44, height: 36, borderRadius: 7, overflow: 'hidden',
                          border: isOnSite ? '1.5px solid #ef4444' : '1.5px solid #cbd5e1',
                          background: '#fff5f5', flexShrink: 0, display: 'flex',
                          alignItems: 'center', justifyContent: 'center'
                        }}
                        title={amb.photo_url ? 'Click to view ambulance capture' : 'No snapshot'}
                      >
                        {amb.photo_url ? (
                          <img src={`${API}${amb.photo_url}`} alt={plateStr} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        ) : (
                          <span style={{ fontSize: 17 }}>🚑</span>
                        )}
                      </div>
                      <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                          {amb.plate ? (
                            <span style={{ fontFamily: 'var(--font-mono, monospace)', fontWeight: 800, fontSize: 12, color: '#0f172a', background: '#f1f5f9', padding: '2px 7px', borderRadius: 5, border: '1px solid #cbd5e1' }}>
                              {amb.plate}
                            </span>
                          ) : (
                            <span style={{ fontSize: 10, fontWeight: 800, color: '#9a3412', background: '#ffedd5', padding: '2px 7px', borderRadius: 5, border: '1px solid #fdba74' }}>
                              PLATE NOT READ
                            </span>
                          )}
                          <span style={{
                            fontSize: 9, fontWeight: 800, padding: '2px 6px', borderRadius: 4,
                            background: isOnSite ? '#fee2e2' : '#dcfce7',
                            color: isOnSite ? '#dc2626' : '#15803d',
                            border: `1px solid ${isOnSite ? '#fca5a5' : '#86efac'}`
                          }}>
                            {isOnSite ? '● ON PREMISES' : '✓ EXITED'}
                          </span>
                        </div>
                        <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
                          {(amb.cameras || []).map(c => c.name || c.stream).join(' → ') || 'Gate Detection'}
                        </div>
                      </div>
                    </div>
                  </td>

                  <td style={{ whiteSpace: 'nowrap' }}>
                    {amb.entry ? (
                      <div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: '#166534' }}>{tsDate(amb.entry.at)}</div>
                        <div style={{ fontSize: 10, color: '#64748b' }}>{amb.entry.device || amb.entry.stream}</div>
                      </div>
                    ) : (
                      <span className="muted" style={{ fontSize: 10 }}>Detected at Exit only</span>
                    )}
                  </td>

                  <td style={{ whiteSpace: 'nowrap' }}>
                    {isOnSite ? (
                      <div>
                        <span style={{ fontSize: 11, fontWeight: 800, color: '#ea580c', background: '#fff7ed', padding: '2px 8px', borderRadius: 5, border: '1px solid #fed7aa' }}>
                          🔴 Still On Premises
                        </span>
                        <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>Has not exited gate yet</div>
                      </div>
                    ) : (
                      <div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: '#1d4ed8' }}>{tsDate(amb.exit?.at || amb.last_at)}</div>
                        <div style={{ fontSize: 10, color: '#64748b' }}>
                          {amb.exit_confirmed ? (amb.exit?.device || 'Exit Gate') : 'Inferred Exit (No Exit Read)'}
                        </div>
                      </div>
                    )}
                  </td>

                  <td style={{ whiteSpace: 'nowrap' }}>
                    {amb.source === 'journey' ? (
                      <span style={{ padding: '3px 8px', borderRadius: 6, background: '#f1f5f9', color: '#0f172a', fontWeight: 800, fontSize: 11 }}>
                        {dwell(amb.dwell_ms)}
                      </span>
                    ) : (
                      <span style={{ padding: '3px 8px', borderRadius: 6, background: '#fff7ed', color: '#9a3412', fontWeight: 700, fontSize: 10 }}>
                        {amb.direction === 'incoming' ? 'Gate Entry' : 'Gate Exit'}
                      </span>
                    )}
                  </td>

                  <td>
                    <div style={{ fontSize: 11, fontWeight: 700, color: '#334155' }}>
                      {amb.reads || 1} read{(amb.reads || 1) !== 1 ? 's' : ''}
                      {amb.confidence > 0 && ` • ${Math.round(amb.confidence * 100)}%`}
                    </div>
                    {amb.last_dispatch ? (
                      <div
                        style={{
                          display: 'inline-flex', alignItems: 'center', gap: 4, marginTop: 3,
                          padding: '1px 6px', borderRadius: 4, background: '#dcfce7', color: '#166534',
                          border: '1px solid #86efac', fontSize: 9, fontWeight: 800
                        }}
                        title={`Last emailed to ${amb.last_dispatch.recipients} at ${tsDate(amb.last_dispatch.ts)}`}
                      >
                        ✓ Emailed • {ts(amb.last_dispatch.ts)}
                      </div>
                    ) : (
                      <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 2 }}>No alert email sent yet</div>
                    )}
                  </td>

                  <td style={{ textAlign: 'right', whiteSpace: 'nowrap' }}>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      {amb.photo_url && (
                        <button
                          type="button"
                          className="btn sm"
                          onClick={() => onOpenSightingPhoto({
                            photo_url: amb.photo_url,
                            plate: plateStr,
                            stream: amb.entry?.stream || amb.exit?.stream || 'ambulance',
                            ts: amb.last_at || amb.first_at,
                            confidence: amb.confidence
                          })}
                          style={{ fontSize: 11, padding: '4px 9px' }}
                          title="Inspect captured ambulance snapshot"
                        >
                          📷 Photo
                        </button>
                      )}
                      <button
                        type="button"
                        onClick={() => onOpenDispatch(buildDispatchPayload(amb))}
                        style={{
                          padding: '4px 11px', borderRadius: 6, backgroundColor: '#fff7ed',
                          border: '1px solid #fdba74', fontSize: 11, cursor: 'pointer',
                          color: '#c2410c', fontWeight: 800, display: 'inline-flex',
                          alignItems: 'center', gap: 4
                        }}
                        title="Dispatch AWS SES email alert for this ambulance"
                      >
                        ✉ Dispatch Email
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="empty">
                  No ambulances detected in this selected time window.
                </td>
              </tr>
            )}
          </tbody>
        </table>
        </div>
      </div>
    </div>
  );
}

/* ── Journey Rules Panel ──────────────────────────────────── */

function RulesPanel({ rules, onUpdate }) {
  const [visitGap, setVisitGap] = useState(rules?.visit_gap_min ?? 30);
  const [exitGrace, setExitGrace] = useState(rules?.exit_grace_min ?? 45);
  const [correlateWindow, setCorrelateWindow] = useState(rules?.correlate_window_s ?? 300);
  const [gateDedupS, setGateDedupS] = useState(rules?.gate_dedup_s ?? 15);
  const [correlatePassages, setCorrelatePassages] = useState(rules?.correlate_passages ?? true);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (rules) {
      setVisitGap(rules.visit_gap_min);
      setExitGrace(rules.exit_grace_min);
      setCorrelateWindow(rules.correlate_window_s);
      setGateDedupS(rules.gate_dedup_s ?? 15);
      setCorrelatePassages(rules.correlate_passages);
    }
  }, [rules]);

  const save = async () => {
    setBusy(true);
    try {
      await api('/rules', {
        visit_gap_min: visitGap,
        exit_grace_min: exitGrace,
        correlate_window_s: correlateWindow,
        gate_dedup_s: gateDedupS,
        correlate_passages: correlatePassages
      });
      onUpdate();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="field" style={{ marginBottom: 14 }}>
        <span>Visit Gap (minutes)</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <input type="range" min={1} max={120} value={visitGap} onChange={e => setVisitGap(Number(e.target.value))} style={{ flex: 1 }} />
          <b style={{ fontSize: 12, minWidth: 30 }}>{visitGap}m</b>
        </div>
        <small style={{ color: '#8ca0bc', fontSize: 10 }}>Minutes of silence before a new visit starts for the same plate</small>
      </div>
      <div className="field" style={{ marginBottom: 14 }}>
        <span>Exit Grace (minutes)</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <input type="range" min={1} max={120} value={exitGrace} onChange={e => setExitGrace(Number(e.target.value))} style={{ flex: 1 }} />
          <b style={{ fontSize: 12, minWidth: 30 }}>{exitGrace}m</b>
        </div>
        <small style={{ color: '#8ca0bc', fontSize: 10 }}>Minutes after last sighting to consider vehicle exited</small>
      </div>
      <div className="field" style={{ marginBottom: 14 }}>
        <span>Correlate Window (seconds)</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <input type="range" min={30} max={3600} step={30} value={correlateWindow} onChange={e => setCorrelateWindow(Number(e.target.value))} style={{ flex: 1 }} />
          <b style={{ fontSize: 12, minWidth: 40 }}>{correlateWindow}s</b>
        </div>
      </div>
      <div className="field" style={{ marginBottom: 14 }}>
        <span>Multi-Camera Gate Dedup Window (seconds)</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <input type="range" min={0} max={120} step={1} value={gateDedupS} onChange={e => setGateDedupS(Number(e.target.value))} style={{ flex: 1 }} />
          <b style={{ fontSize: 12, minWidth: 36 }}>{gateDedupS}s</b>
        </div>
        <small style={{ color: '#8ca0bc', fontSize: 10 }}>
          Merges overlapping Entry/Exit cameras watching one physical gate (0s = exact plate only)
        </small>
      </div>
      <label className="check">
        <input type="checkbox" checked={correlatePassages} onChange={e => setCorrelatePassages(e.target.checked)} />
        Correlate Passages
      </label>
      <button className="btn primary" style={{ width: '100%', marginTop: 8 }} onClick={save} disabled={busy}>
        {busy ? 'Updating…' : 'Update Journey Rules'}
      </button>
    </div>
  );
}

/* ── Excel Export Helper ──────────────────────────────────── */

function exportJourneysToExcel({ journeys, apiOrigin }) {
  const wb = XLSX.utils.book_new();

  // Clean table headers starting directly on Row 1 — NO metadata or summary lines!
  const tableHeaders = [
    'S.No',
    'License Plate',
    'Vehicle Type',
    'Status',
    'Entry Camera',
    'Entry Date & Time',
    'Exit Camera',
    'Exit Date & Time',
    'Dwell Duration',
    'Cameras Visited',
    'Total Reads',
    'Entry Vehicle Photo',
    'Exit Vehicle Photo',
  ];

  const dataRows = journeys.map((j, idx) => {
    const entryTime = j.first_at ? new Date(j.first_at).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }) : '—';
    const exitTime = j.exit?.at ? new Date(j.exit.at).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }) : '—';
    const path = (j.cameras || []).map(c => c.name).join(' → ') || j.entry?.device || '—';
    const entryPhotoUrl = j.entry?.photo_url ? `${apiOrigin}${j.entry.photo_url}` : null;
    const exitPhotoUrl = j.exit?.photo_url ? `${apiOrigin}${j.exit.photo_url}` : null;

    return [
      idx + 1,
      j.plate,
      String(j.vehicle_type || 'car').toUpperCase(),
      j.status === 'on_site' ? 'ON SITE' : 'EXITED',
      j.entry?.device || '—',
      entryTime,
      j.exit?.device || '—',
      exitTime,
      dwell(j.dwell_ms),
      path,
      j.reads || 1,
      entryPhotoUrl ? { t: 's', v: 'View Entry Photo', l: { Target: entryPhotoUrl, Tooltip: 'Click to open vehicle & plate photo in browser' } } : '—',
      exitPhotoUrl ? { t: 's', v: 'View Exit Photo', l: { Target: exitPhotoUrl, Tooltip: 'Click to open vehicle & plate photo in browser' } } : '—',
    ];
  });

  const ws = XLSX.utils.aoa_to_sheet([tableHeaders, ...dataRows]);

  ws['!cols'] = [
    { wch: 6 },  // S.No
    { wch: 16 }, // License Plate
    { wch: 14 }, // Vehicle Type
    { wch: 12 }, // Status
    { wch: 26 }, // Entry Camera
    { wch: 22 }, // Entry Date & Time
    { wch: 26 }, // Exit Camera
    { wch: 22 }, // Exit Date & Time
    { wch: 16 }, // Dwell Duration
    { wch: 36 }, // Cameras Visited
    { wch: 12 }, // Total Reads
    { wch: 20 }, // Entry Vehicle Photo
    { wch: 20 }, // Exit Vehicle Photo
  ];

  XLSX.utils.book_append_sheet(wb, ws, 'Vehicle Journeys');

  const d = new Date();
  const dateStr = d.toISOString().slice(0, 10);
  const timeStr = `${String(d.getHours()).padStart(2, '0')}-${String(d.getMinutes()).padStart(2, '0')}`;
  const filename = `Vehicle_Journeys_${dateStr}_${timeStr}.xlsx`;

  XLSX.writeFile(wb, filename);
}


/* ── Main App ─────────────────────────────────────────────── */

function App() {
  const [clockTime, setClockTime] = useState(() => {
    return new Date().toLocaleTimeString('en-IN', {
      timeZone: 'Asia/Kolkata',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    }).toUpperCase() + ' IST';
  });

  useEffect(() => {
    const timer = setInterval(() => {
      setClockTime(new Date().toLocaleTimeString('en-IN', {
        timeZone: 'Asia/Kolkata',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
      }).toUpperCase() + ' IST');
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const [feedSource, setFeedSource] = useState(null);
  const [state, setState] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [todayAnalytics, setTodayAnalytics] = useState(null);
  const [todayVehicleCounts, setTodayVehicleCounts] = useState(null);
  const [journeyData, setJourneyData] = useState([]);
  const [expandedJourney, setExpandedJourney] = useState(null);
  const [addOpen, setAddOpen] = useState(false);
  const [editDevice, setEditDevice] = useState(null);
  const [searchPlate, setSearchPlate] = useState('');
  const [debouncedSearchPlate, setDebouncedSearchPlate] = useState('');
  useEffect(() => {
    const trimmed = searchPlate.trim();
    if (!trimmed) {
      setDebouncedSearchPlate('');
      return;
    }
    const tid = setTimeout(() => setDebouncedSearchPlate(trimmed), 300);
    return () => clearTimeout(tid);
  }, [searchPlate]);
  const [timeFilter, setTimeFilter] = useState('1h'); // '15m' | '30m' | '1h' | '2h' | '6h' | '12h' | '24h' | 'today' | 'custom'
  const [customStart, setCustomStart] = useState(() => {
    const d = new Date();
    d.setHours(d.getHours() - 1);
    return d.toISOString().slice(0, 16);
  });
  const [customEnd, setCustomEnd] = useState(() => {
    return new Date().toISOString().slice(0, 16);
  });
  const [selectedCameraFilter, setSelectedCameraFilter] = useState('all');
  const [selectedStatusFilter, setSelectedStatusFilter] = useState('all');
  const [selectedTypeFilter, setSelectedTypeFilter] = useState('all');
  const [pageSize, setPageSize] = useState(25);
  const [currentPage, setCurrentPage] = useState(1);
  const [exportNotice, setExportNotice] = useState(null);
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState(null);

  // Dynamic View Grid slots (array of stream IDs e.g. ['mgptz', 'exit_gate'])
  const [activeSlots, setActiveSlots] = useState(() => {
    try {
      const saved = localStorage.getItem('vehicle_tracker_view_slots');
      if (saved) return JSON.parse(saved);
    } catch { }
    return ['mgptz', 'mg_outside', 'exit_gate', 'egtmp'];
  });


  // Modals
  const [fullscreenStream, setFullscreenStream] = useState(null);
  const [selectedSightingModal, setSelectedSightingModal] = useState(null);
  const [vehicleCounts, setVehicleCounts] = useState(null);
  const [vcRange, setVcRange] = useState('today');
  const [vcCustomStart, setVcCustomStart] = useState(() => {
    const d = new Date(Date.now() - 86400000);
    return d.toISOString().slice(0, 16);
  });
  const [vcCustomEnd, setVcCustomEnd] = useState(() => new Date().toISOString().slice(0, 16));
  const [historyScope, setHistoryScope] = useState(null);
  const [ambulanceData, setAmbulanceData] = useState(null);
  const [ambHours, setAmbHours] = useState('24');
  const [dispatchAmbulance, setDispatchAmbulance] = useState(null);
  const [vcLoading, setVcLoading] = useState(true);
  const [jtLoading, setJtLoading] = useState(true);
  const [ambLoading, setAmbLoading] = useState(true);

  const vcReqRef = useRef(0);
  const jtReqRef = useRef(0);
  const ambReqRef = useRef(0);
  const pollTickRef = useRef(0);

  const devices = state?.devices || [];
  const rules = state?.rules;
  const engine = state?.engine;

  const deviceMap = useMemo(() => {
    const m = {};
    (devices || []).forEach(d => { m[d.stream] = d; });
    return m;
  }, [devices]);

  // Save viewSlots to localStorage
  const updateActiveSlots = useCallback((newSlots) => {
    setActiveSlots(newSlots);
    try {
      localStorage.setItem('vehicle_tracker_view_slots', JSON.stringify(newSlots));
    } catch { }
  }, []);

  // Presets
  const setPreset = useCallback((preset) => {
    if (preset === '1-focus') {
      updateActiveSlots(['mgptz']);
    } else if (preset === 'entry-exit') {
      updateActiveSlots(['mgptz', 'exit_gate']);
    } else if (preset === 'entry-pair') {
      updateActiveSlots(['mgptz', 'mg_outside']);
    } else if (preset === 'exit-pair') {
      updateActiveSlots(['exit_gate', 'egtmp']);
    } else if (preset === 'all-4') {
      updateActiveSlots(['mgptz', 'mg_outside', 'exit_gate', 'egtmp']);
    }
  }, [updateActiveSlots]);

  const handleSlotChange = useCallback((slotIndex, stream) => {
    const next = [...activeSlots];
    if (stream) {
      next[slotIndex] = stream;
    } else {
      next[slotIndex] = null;
    }
    updateActiveSlots(next);
  }, [activeSlots, updateActiveSlots]);

  const handleRemoveSlot = useCallback((slotIndex) => {
    if (activeSlots.length <= 1) return;
    const next = activeSlots.filter((_, idx) => idx !== slotIndex);
    updateActiveSlots(next);
  }, [activeSlots, updateActiveSlots]);

  const handleAddSlot = useCallback(() => {
    // Pick the first registered camera that is not yet visible in activeSlots
    const inUse = new Set(activeSlots.filter(Boolean));
    const nextAvailable = devices.find(d => !inUse.has(d.stream));
    const toAdd = nextAvailable ? nextAvailable.stream : (devices[0]?.stream || null);
    updateActiveSlots([...activeSlots, toAdd]);
  }, [activeSlots, devices, updateActiveSlots]);

  const handleToggleViewSlot = useCallback((stream) => {
    if (activeSlots.includes(stream)) {
      if (activeSlots.length > 1) {
        updateActiveSlots(activeSlots.filter(s => s !== stream));
      }
    } else {
      updateActiveSlots([...activeSlots, stream]);
    }
  }, [activeSlots, updateActiveSlots]);

  // Data fetching with per-section filter loading states & stale-response guards
  const fetchTrafficAndJourneys = useCallback(async (showLoader = false) => {
    const reqId = ++jtReqRef.current;
    if (showLoader) setJtLoading(true);
    try {
      let jtWindowQs = 'hours=24';
      if (timeFilter === '15m') jtWindowQs = 'hours=0.25';
      else if (timeFilter === '30m') jtWindowQs = 'hours=0.5';
      else if (timeFilter === '1h') jtWindowQs = 'hours=1';
      else if (timeFilter === '2h') jtWindowQs = 'hours=2';
      else if (timeFilter === '6h') jtWindowQs = 'hours=6';
      else if (timeFilter === '12h') jtWindowQs = 'hours=12';
      else if (timeFilter === '24h') jtWindowQs = 'hours=24';
      else if (timeFilter === 'today') jtWindowQs = 'hours=today';
      else if (timeFilter === '7d') jtWindowQs = 'hours=168';
      else if (timeFilter === 'custom') {
        const fMs = new Date(customStart).getTime() || 0;
        const tMs = new Date(customEnd).getTime() || Date.now();
        jtWindowQs = `from_ts=${fMs}&to_ts=${tMs}`;
      }
      const [analyticsData, journeyRes] = await Promise.all([
        api('/analytics?' + jtWindowQs),
        api('/journeys?' + jtWindowQs + (debouncedSearchPlate ? '&plate=' + encodeURIComponent(debouncedSearchPlate) : ''))
      ]);
      if (reqId === jtReqRef.current) {
        setAnalytics(analyticsData);
        setJourneyData(journeyRes.journeys || []);
        setError(null);
      }
    } catch (err) {
      if (reqId === jtReqRef.current) setError(err.message);
    } finally {
      if (reqId === jtReqRef.current) setJtLoading(false);
    }
  }, [timeFilter, customStart, customEnd, debouncedSearchPlate]);

  const fetchVehicleCounts = useCallback(async (showLoader = false) => {
    const reqId = ++vcReqRef.current;
    if (showLoader) setVcLoading(true);
    try {
      let vcQuery = 'hours=today';
      if (vcRange === '1h') vcQuery = 'hours=1';
      else if (vcRange === '24h') vcQuery = 'hours=24';
      else if (vcRange === '7d') vcQuery = 'hours=168';
      else if (vcRange === 'all') vcQuery = 'hours=all';
      else if (vcRange === 'custom') {
        const fMs = new Date(vcCustomStart).getTime() || 0;
        const tMs = new Date(vcCustomEnd).getTime() || Date.now();
        vcQuery = `from_ts=${fMs}&to_ts=${tMs}`;
      }
      const vcRes = await api('/vehicle-counts?' + vcQuery).catch(() => null);
      if (reqId === vcReqRef.current && vcRes) setVehicleCounts(vcRes);
    } finally {
      if (reqId === vcReqRef.current) setVcLoading(false);
    }
  }, [vcRange, vcCustomStart, vcCustomEnd]);

  const fetchAmbulances = useCallback(async (showLoader = false) => {
    const reqId = ++ambReqRef.current;
    if (showLoader) setAmbLoading(true);
    try {
      const ambHoursParam = ambHours === 'today' ? 'today' : ambHours;
      const ambRes = await api('/ambulances?hours=' + ambHoursParam).catch(() => null);
      if (reqId === ambReqRef.current && ambRes) setAmbulanceData(ambRes);
    } finally {
      if (reqId === ambReqRef.current) setAmbLoading(false);
    }
  }, [ambHours]);

  const fetchTodayKpis = useCallback(async () => {
    try {
      const [todayAn, todayVc] = await Promise.all([
        api('/analytics?hours=today').catch(() => null),
        api('/vehicle-counts?hours=today').catch(() => null),
      ]);
      if (todayAn) setTodayAnalytics(todayAn);
      if (todayVc) setTodayVehicleCounts(todayVc);
    } catch { }
  }, []);

  const pollAll = useCallback(async (showLoaders = false, forceAll = false) => {
    try {
      const isBgTick = !showLoaders && !forceAll;
      if (isBgTick && typeof document !== 'undefined' && document.hidden) return;
      const tick = isBgTick ? (++pollTickRef.current) : 0;
      const isJtHeavy = (timeFilter === '7d' || timeFilter === 'custom');
      const isVcHeavy = (vcRange === '7d' || vcRange === 'all' || vcRange === 'custom');
      const isAmbHeavy = (String(ambHours) === '168');

      const tasks = [api('/state'), fetchTodayKpis()];
      if (!isBgTick || !isJtHeavy || (tick % 8 === 0)) tasks.push(fetchTrafficAndJourneys(showLoaders));
      if (!isBgTick || !isVcHeavy || (tick % 8 === 0)) tasks.push(fetchVehicleCounts(showLoaders));
      if (!isBgTick || !isAmbHeavy || (tick % 8 === 0)) tasks.push(fetchAmbulances(showLoaders));

      const [stateData] = await Promise.all(tasks);
      if (stateData?.feed_mode) setFeedSource(stateData.feed_mode);
      setState(stateData);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, [fetchTodayKpis, fetchTrafficAndJourneys, fetchVehicleCounts, fetchAmbulances, timeFilter, vcRange, ambHours]);

  const loadState = useCallback(() => pollAll(false, true), [pollAll]);

  const handleToggleFeedSource = async () => {
    const nextMode = feedSource === 'demo' ? 'live' : 'demo';
    try {
      const res = await api('/feed-source', { mode: nextMode });
      if (res?.mode) {
        setFeedSource(res.mode);
        pollAll(true);
      }
    } catch (e) {
      console.error("Toggle feed source error:", e);
    }
  };

  const pollSilentRef = useRef(null);
  useEffect(() => {
    pollSilentRef.current = () => pollAll(false);
  }, [pollAll]);

  useEffect(() => {
    pollAll(true);
    const iv = setInterval(() => {
      if (pollSilentRef.current) pollSilentRef.current();
    }, POLL_INTERVAL);
    return () => clearInterval(iv);
  }, []);

  const jtMountRef = useRef(false);
  useEffect(() => {
    if (!jtMountRef.current) { jtMountRef.current = true; return; }
    fetchTrafficAndJourneys(true);
  }, [fetchTrafficAndJourneys]);

  const vcMountRef = useRef(false);
  useEffect(() => {
    if (!vcMountRef.current) { vcMountRef.current = true; return; }
    fetchVehicleCounts(true);
  }, [fetchVehicleCounts]);

  const ambMountRef = useRef(false);
  useEffect(() => {
    if (!ambMountRef.current) { ambMountRef.current = true; return; }
    fetchAmbulances(true);
  }, [fetchAmbulances]);

  // Filter journeys in real-time in front of the user
  const filteredJourneys = useMemo(() => {
    const now = Date.now();
    let minTs = 0;
    let maxTs = Infinity;

    if (timeFilter === '15m') minTs = now - 15 * 60 * 1000;
    else if (timeFilter === '30m') minTs = now - 30 * 60 * 1000;
    else if (timeFilter === '1h') minTs = now - 60 * 60 * 1000;
    else if (timeFilter === '2h') minTs = now - 2 * 60 * 60 * 1000;
    else if (timeFilter === '6h') minTs = now - 6 * 60 * 60 * 1000;
    else if (timeFilter === '12h') minTs = now - 12 * 60 * 60 * 1000;
    else if (timeFilter === '24h') minTs = now - 24 * 60 * 60 * 1000;
    else if (timeFilter === '7d') minTs = now - 7 * 24 * 60 * 60 * 1000;
    else if (timeFilter === 'today') {
      const istDateStr = new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Kolkata' }); // YYYY-MM-DD in IST
      minTs = new Date(`${istDateStr}T00:00:00+05:30`).getTime();
    } else if (timeFilter === 'custom') {
      minTs = new Date(customStart).getTime() || 0;
      maxTs = new Date(customEnd).getTime() || Infinity;
    }

    return (journeyData || []).filter(j => {
      // Time filter
      if (j.first_at < minTs || j.first_at > maxTs) return false;

      // Status filter
      if (selectedStatusFilter !== 'all' && j.status !== selectedStatusFilter) {
        return false;
      }

      // Vehicle type filter
      const jType = String(j.vehicle_type || 'car').toLowerCase();
      if (selectedTypeFilter !== 'all' && jType !== selectedTypeFilter) {
        return false;
      }

      // Camera filter
      if (selectedCameraFilter === 'role_entry') {
        const matches = (j.entry?.role === 'entry') || (j.cameras || []).some(c => c.role === 'entry');
        if (!matches) return false;
      } else if (selectedCameraFilter === 'role_exit') {
        const matches = (j.exit?.role === 'exit') || (j.cameras || []).some(c => c.role === 'exit');
        if (!matches) return false;
      } else if (selectedCameraFilter === 'role_checkpoint') {
        const matches = (j.cameras || []).some(c => c.role === 'checkpoint' || (c.role !== 'entry' && c.role !== 'exit'));
        if (!matches) return false;
      } else if (selectedCameraFilter !== 'all') {
        const matchesCam = (j.cameras || []).some(c => c.stream === selectedCameraFilter || c.name === selectedCameraFilter)
          || j.entry?.stream === selectedCameraFilter
          || j.exit?.stream === selectedCameraFilter;
        if (!matchesCam) return false;
      }

      // Search plate filter
      if (searchPlate && !j.plate.includes(searchPlate.toUpperCase().trim())) {
        return false;
      }

      return true;
    });
  }, [journeyData, timeFilter, customStart, customEnd, selectedStatusFilter, selectedTypeFilter, selectedCameraFilter, searchPlate]);

  // Download ONLY the filtered list in front of the user
  const handleExportExcel = () => {
    if (!filteredJourneys || filteredJourneys.length === 0) {
      alert('No vehicle journeys match the selected criteria to export.');
      return;
    }
    setIsExporting(true);
    try {
      const apiOrigin = window.location.port === '8190' ? 'http://124.123.89.162:31016' : window.location.origin;

      exportJourneysToExcel({
        journeys: filteredJourneys,
        apiOrigin
      });

      setExportNotice(`✅ Downloaded Excel: ${filteredJourneys.length} filtered vehicles!`);
      setTimeout(() => setExportNotice(null), 4000);
    } catch (err) {
      alert('Error exporting Excel: ' + err.message);
    } finally {
      setIsExporting(false);
    }
  };

  const handleClearFilters = () => {
    setSelectedCameraFilter('all');
    setSelectedStatusFilter('all');
    setSelectedTypeFilter('all');
    setSearchPlate('');
    setTimeFilter('1h');
  };

  // Reset page when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [selectedCameraFilter, selectedStatusFilter, selectedTypeFilter, searchPlate, timeFilter, pageSize]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(filteredJourneys.length / pageSize));
  const paginatedJourneys = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredJourneys.slice(start, start + pageSize);
  }, [filteredJourneys, currentPage, pageSize]);

  // Dynamic video height and grid template columns based on number of active camera slots
  const slotCount = activeSlots.length;
  let gridColumns = `repeat(auto-fit, minmax(280px, 1fr))`;
  let videoHeight = 220;

  if (slotCount === 1) {
    gridColumns = '1fr';
    videoHeight = 440; // Full enlargement when viewing only 1 camera
  } else if (slotCount === 2) {
    gridColumns = 'repeat(2, 1fr)';
    videoHeight = 330; // Medium enlargement when viewing 2 side-by-side
  } else if (slotCount === 3) {
    gridColumns = 'repeat(3, 1fr)';
    videoHeight = 260;
  } else if (slotCount >= 4) {
    gridColumns = 'repeat(4, 1fr)';
    videoHeight = 220;
  }

  return (
    <div className="shell">
      {/* ── Top Bar ── */}
      <div className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="1" y="3" width="15" height="13" rx="2" /><polygon points="23 7 16 12 16 7" /><path d="M3 19h18" /><circle cx="7" cy="19" r="2" /><circle cx="17" cy="19" r="2" />
            </svg>
          </div>
          <div>
            <h1>Medanta Operations</h1>
            <p>Hospital AI Suite • Vehicle Journey & ANPR Engine</p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <button
            type="button"
            onClick={handleToggleFeedSource}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '7px',
              padding: '6px 14px',
              borderRadius: '8px',
              fontSize: '12px',
              fontWeight: '700',
              cursor: 'pointer',
              border: feedSource === 'demo' ? '1.5px solid #8b5cf6' : '1.5px solid #10b981',
              backgroundColor: feedSource === 'demo' ? '#f5f3ff' : '#ecfdf5',
              color: feedSource === 'demo' ? '#6d28d9' : '#047857',
              boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
              transition: 'all 0.2s ease'
            }}
            title={feedSource === 'demo' ? "Currently playing Demo Recordings. Click to switch to Live RTSP Feeds." : "Currently streaming Live RTSP Feeds. Click to switch to Demo Recordings."}
          >
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: feedSource === 'demo' ? '#7c3aed' : '#10b981',
              boxShadow: feedSource === 'demo' ? '0 0 6px #7c3aed' : '0 0 6px #10b981'
            }}></span>
            <span>{feedSource === 'demo' ? '🎬 Demo Mode (Recordings)' : '🟢 Live Feeds Active'}</span>
            <span style={{ fontSize: '10px', opacity: 0.7, textDecoration: 'underline', marginLeft: '2px' }}>
              (Switch)
            </span>
          </button>
          <span className={`status-pill ${engine?.running ? 'live' : 'down'}`}>
            <span className="dot" />
            GPU ANPR Engine • {engine?.running ? 'ACTIVE' : 'OFFLINE'}
          </span>
          <span style={{ fontSize: 12, fontWeight: 700, color: engine?.running ? '#22a06e' : '#c04848' }}>
            🟢 {clockTime}
          </span>
        </div>
      </div>

      {error && <div className="notice err">⚠ Connection error: {error}</div>}
      {engine?.model_error && <div className="notice err">⚠ Model error: {engine.model_error}</div>}

      {/* ── KPI Cards (Always locked to Today's IST data regardless of section filters) ── */}
      <div className="kpis">
        <KpiCard label="VEHICLES TODAY" value={todayAnalytics?.vehicles_today ?? '—'} sub="Today's unique entries" color="var(--teal)" />
        <KpiCard label="ON SITE NOW" value={todayAnalytics?.on_site ?? '—'} sub="Currently in premises" color="var(--blue)" />
        <BasementInKpiCard vehicleCounts={todayVehicleCounts} loading={!todayVehicleCounts} />
        <KpiCard label="UNIQUE PLATES" value={todayAnalytics?.unique_plates ?? '—'} sub="Today's unique plates" color="var(--purple)" />
        <KpiCard label="AVG DWELL TIME" value={todayAnalytics?.avg_dwell_ms ? dwell(todayAnalytics.avg_dwell_ms) : '—'} sub="Today's entry to exit" color="var(--gold)" />
      </div>

      {/* ── Live Camera Grid Card with Dynamic Add/Remove & Auto-Enlarge ── */}
      <div className="card span12" style={{ marginBottom: 16 }}>
        {/* Head Bar */}
        <div className="head" style={{ flexWrap: 'wrap', gap: 12, padding: '14px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <h3 style={{ margin: 0 }}>
              Live Vehicle Camera Feeds ({slotCount}-Slot Grid)
            </h3>
            <span className="badge" style={{ background: '#e3f6ef', color: '#17835c', fontSize: 9 }}>
              ⚡ Inference running 24/7 on all {devices.length} cameras
            </span>
          </div>

          {/* Quick Presets & Controls */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 10, fontWeight: 700, color: '#8ca0bc', textTransform: 'uppercase' }}>Presets:</span>
            <button
              className={`preset-btn ${slotCount === 1 ? 'active' : ''}`}
              onClick={() => setPreset('1-focus')}
              title="Focus on single camera in large format"
            >
              1 Feed (Large)
            </button>
            <button
              className={`preset-btn ${slotCount === 2 && activeSlots.includes('mgptz') && activeSlots.includes('exit_gate') ? 'active' : ''}`}
              onClick={() => setPreset('entry-exit')}
              title="1 Entry + 1 Exit side by side"
            >
              1 Entry + 1 Exit
            </button>
            <button
              className={`preset-btn ${slotCount === 2 && activeSlots.includes('mgptz') && activeSlots.includes('mg_outside') ? 'active' : ''}`}
              onClick={() => setPreset('entry-pair')}
              title="Both Entry Cameras side by side"
            >
              Entry Pair
            </button>
            <button
              className={`preset-btn ${slotCount === 2 && activeSlots.includes('exit_gate') && activeSlots.includes('egtmp') ? 'active' : ''}`}
              onClick={() => setPreset('exit-pair')}
              title="Both Exit Cameras side by side"
            >
              Exit Pair
            </button>
            <button
              className={`preset-btn ${slotCount >= 4 ? 'active' : ''}`}
              onClick={() => setPreset('all-4')}
              title="All 4 camera feeds"
            >
              All 4 Feeds
            </button>
            <button
              className="btn sm"
              onClick={handleAddSlot}
              style={{ background: '#206e69', color: '#fff', borderColor: '#206e69', fontWeight: 700 }}
              title="Add another camera slot to grid"
            >
              + Add Slot
            </button>
            <button className="btn sm primary" onClick={() => setAddOpen(true)}>
              + Register Camera
            </button>
          </div>
        </div>

        {/* Dynamic Video Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: gridColumns,
          gap: 12,
          padding: 14,
          background: 'linear-gradient(180deg, #0f1925, #101e2e)',
          borderRadius: '0 0 14px 14px'
        }}>
          {activeSlots.map((stream, i) => (
            <CameraSlot
              key={i + '-' + (stream || 'empty') + '-' + feedSource}
              slotIndex={i}
              stream={stream}
              devices={devices}
              activeSlots={activeSlots}
              onChangeSlot={handleSlotChange}
              onRemoveSlot={handleRemoveSlot}
              onFullscreen={(s) => setFullscreenStream(s)}
              canRemove={activeSlots.length > 1}
              engineWorkers={engine?.workers}
              videoHeight={videoHeight}
              feedSource={feedSource}
            />
          ))}
        </div>
      </div>

      {/* ── Main Content Grid ── */}
      <div className="grid">
        {/* Left: Gate Checkpoints + Journeys + Inflow Volume Timeline + Remaining Live Cameras */}
        <div className="span8" style={{ display: 'grid', gap: 16, alignContent: 'start' }}>
          {/* ── Gate Traffic & Camera Flow Analysis Card (Checkpoints) ── */}
          <GateAnalyticsCard
            analytics={analytics}
            devices={devices}
            timeFilter={timeFilter}
            setTimeFilter={setTimeFilter}
            loading={jtLoading}
          />

          {/* Journey Table Card */}
          <div className="card">
            {/* Header: Title & Download Filtered List Button */}
            <div className="head" style={{ flexWrap: 'wrap', gap: 10, padding: '14px 18px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <h3 style={{ margin: 0, fontSize: 16 }}>🚗 Vehicle Journeys</h3>
                <span className="badge" style={{ background: '#eef2f9', color: '#37465e', fontSize: 11, fontWeight: 700 }}>
                  {filteredJourneys.length} matching filter{filteredJourneys.length !== 1 ? 's' : ''}
                </span>
                <LoadingPill show={jtLoading} text="Loading…" />
                {exportNotice && (
                  <span className="badge" style={{ background: '#e3f6ef', color: '#17835c', fontSize: 11, fontWeight: 700 }}>
                    {exportNotice}
                  </span>
                )}
              </div>

              {/* Prominent Download Button */}
              <button
                className="btn sm"
                onClick={handleExportExcel}
                disabled={isExporting || filteredJourneys.length === 0}
                style={{
                  background: '#17835c',
                  color: '#fff',
                  borderColor: '#17835c',
                  fontWeight: 700,
                  fontSize: 11,
                  padding: '7px 16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  cursor: filteredJourneys.length > 0 ? 'pointer' : 'not-allowed',
                  boxShadow: '0 2px 10px -2px rgba(23,131,92,0.55)'
                }}
                title="Download this exact filtered list as an Excel spreadsheet with photo links"
              >
                {isExporting ? <span className="spin-ring" /> : <span style={{ fontSize: 13 }}>📥</span>}
                <span>{isExporting ? 'Generating Excel…' : `Download Filtered Excel (${filteredJourneys.length} Vehicles)`}</span>
              </button>
            </div>

            {/* Smart Filters Toolbar — Real-Time in front of the user */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 12, padding: '12px 18px',
              borderBottom: '1px solid #eef2f7', background: '#fafbfc', flexWrap: 'wrap'
            }}>
              {/* Time Filter */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: '#8ca0bc', textTransform: 'uppercase' }}>⏱ Time:</span>
                <select
                  value={timeFilter}
                  onChange={e => setTimeFilter(e.target.value)}
                  style={{ padding: '6px 10px', border: '1px solid #dfe6f0', borderRadius: 7, fontSize: 11, fontWeight: 700, background: '#fff' }}
                >
                  <option value="15m">Last 15 Mins</option>
                  <option value="30m">Last 30 Mins</option>
                  <option value="1h">Last 1 Hour (Default)</option>
                  <option value="2h">Last 2 Hours</option>
                  <option value="6h">Last 6 Hours</option>
                  <option value="12h">Last 12 Hours</option>
                  <option value="24h">Last 24 Hours</option>
                  <option value="today">Today (Since Midnight)</option>
                  <option value="7d">Last 7 Days (Daily Traffic)</option>
                  <option value="custom">📅 Custom Date & Time…</option>
                </select>
              </div>

              {/* Camera Filter */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: '#8ca0bc', textTransform: 'uppercase' }}>📹 Camera:</span>
                <select
                  value={selectedCameraFilter}
                  onChange={e => setSelectedCameraFilter(e.target.value)}
                  style={{ padding: '6px 10px', border: '1px solid #dfe6f0', borderRadius: 7, fontSize: 11, fontWeight: 700, background: '#fff' }}
                >
                  <option value="all">All Cameras ({devices.length})</option>
                  <option value="role_entry">🟢 Any Entry Gate</option>
                  <option value="role_exit">🔵 Any Exit Gate</option>
                  <option value="role_checkpoint">🟣 Any Checkpoint / Porch</option>
                  <optgroup label="Specific Camera">
                    {devices.map(d => (
                      <option key={d.stream} value={d.stream}>
                        📹 {d.name} ({d.role})
                      </option>
                    ))}
                  </optgroup>
                </select>
              </div>

              {/* Status Filter */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: '#8ca0bc', textTransform: 'uppercase' }}>🚦 Status:</span>
                <select
                  value={selectedStatusFilter}
                  onChange={e => setSelectedStatusFilter(e.target.value)}
                  style={{ padding: '6px 10px', border: '1px solid #dfe6f0', borderRadius: 7, fontSize: 11, fontWeight: 700, background: '#fff' }}
                >
                  <option value="all">All Statuses</option>
                  <option value="on_site">🟢 On Site Only</option>
                  <option value="exited">🔵 Exited Only</option>
                </select>
              </div>

              {/* Vehicle Type Filter */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: '#8ca0bc', textTransform: 'uppercase' }}>🚘 Vehicle Type:</span>
                <select
                  value={selectedTypeFilter}
                  onChange={e => setSelectedTypeFilter(e.target.value)}
                  style={{ padding: '6px 10px', border: '1px solid #dfe6f0', borderRadius: 7, fontSize: 11, fontWeight: 700, background: '#fff' }}
                >
                  <option value="all">All Vehicle Types</option>
                  {VEHICLE_TYPE_META.map(vt => (
                    <option key={vt.key} value={vt.key}>{vt.icon} {vt.singular}</option>
                  ))}
                </select>
              </div>

              {/* Plate Search */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <input
                  placeholder="🔍 Plate (e.g. HR26)…"
                  value={searchPlate}
                  onChange={e => setSearchPlate(e.target.value.toUpperCase())}
                  style={{ padding: '6px 10px', border: '1px solid #dfe6f0', borderRadius: 7, fontSize: 11, fontWeight: 600, width: 140 }}
                />
              </div>

              {/* Reset Filters button */}
              {(selectedCameraFilter !== 'all' || selectedStatusFilter !== 'all' || selectedTypeFilter !== 'all' || searchPlate || timeFilter !== '1h') && (
                <button
                  className="btn sm"
                  onClick={handleClearFilters}
                  style={{ fontSize: 10, padding: '5px 10px', color: '#c04848', borderColor: '#f0b4b4', background: '#fff5f5' }}
                >
                  ✕ Reset Filters
                </button>
              )}
            </div>

            {/* Custom Date Range Picker (shown when custom is selected) */}
            {timeFilter === 'custom' && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 10, padding: '8px 18px',
                background: '#f0f9f5', borderBottom: '1px solid #cce8de', flexWrap: 'wrap'
              }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: '#17835c' }}>📅 Custom Date/Time Range:</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 10, color: '#5d7089', fontWeight: 600 }}>From:</span>
                  <input
                    type="datetime-local"
                    value={customStart}
                    onChange={e => setCustomStart(e.target.value)}
                    style={{ border: '1px solid #bfe2d4', borderRadius: 6, fontSize: 11, padding: '4px 6px', fontWeight: 600, background: '#fff' }}
                  />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 10, color: '#5d7089', fontWeight: 600 }}>To:</span>
                  <input
                    type="datetime-local"
                    value={customEnd}
                    onChange={e => setCustomEnd(e.target.value)}
                    style={{ border: '1px solid #bfe2d4', borderRadius: 6, fontSize: 11, padding: '4px 6px', fontWeight: 600, background: '#fff' }}
                  />
                </div>
                <button
                  className="btn sm"
                  onClick={() => {
                    const d = new Date();
                    d.setHours(d.getHours() - 1);
                    setCustomStart(d.toISOString().slice(0, 16));
                    setCustomEnd(new Date().toISOString().slice(0, 16));
                  }}
                  style={{ fontSize: 10, padding: '3px 8px' }}
                >
                  Last 1 Hour
                </button>
                <button
                  className="btn sm"
                  onClick={() => {
                    const d = new Date();
                    d.setHours(0, 0, 0, 0);
                    setCustomStart(d.toISOString().slice(0, 16));
                    setCustomEnd(new Date().toISOString().slice(0, 16));
                  }}
                  style={{ fontSize: 10, padding: '3px 8px' }}
                >
                  Today
                </button>
              </div>
            )}

            {/* List Overview Subbar with Rows Per Page */}
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '8px 18px', background: '#f8fafd', borderBottom: '1px solid #eef2f7',
              fontSize: 11, color: '#5d7089', fontWeight: 600
            }}>
              <div>
                Showing <b>{filteredJourneys.length}</b> vehicles in list
                {filteredJourneys.length !== journeyData.length && (
                  <span style={{ color: '#8ca0bc', marginLeft: 6 }}>
                    (filtered from {journeyData.length} in database)
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span>Rows per page:</span>
                <select
                  value={pageSize}
                  onChange={e => setPageSize(Number(e.target.value))}
                  style={{ padding: '3px 6px', border: '1px solid #dfe6f0', borderRadius: 5, fontSize: 11, fontWeight: 600, background: '#fff' }}
                >
                  <option value={15}>15</option>
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                  <option value={10000}>All</option>
                </select>
              </div>
            </div>

            {/* Compact Scrollable Table Container */}
            <div className="loading-wrap">
              <LoadingOverlay show={jtLoading} text="Loading journeys…" />
              <div style={{ overflowX: 'auto', maxHeight: 520, overflowY: 'auto' }}>
                <table className="tbl">
                  <thead style={{ position: 'sticky', top: 0, background: '#fff', zIndex: 2 }}>
                    <tr>
                      <th>Plate</th>
                      <th>Vehicle Type</th>
                      <th>Status</th>
                      <th>Entry Camera</th>
                      <th>Entry Time</th>
                      <th>Exit Camera</th>
                      <th>Exit Time</th>
                      <th>Dwell</th>
                      <th>Cameras Seen</th>
                      <th style={{ textAlign: 'center' }}>Reads</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedJourneys.map((j, i) => {
                      const jKey = `${j.plate || ''}_${j.first_at || i}`;
                      return (
                        <JourneyFlow
                          key={jKey}
                          journey={j}
                          expanded={expandedJourney === jKey}
                          onToggle={() => setExpandedJourney(prev => (prev === jKey ? null : jKey))}
                          onOpenSightingPhoto={(info) => setSelectedSightingModal(info)}
                        />
                      );
                    })}
                    {filteredJourneys.length === 0 && (
                      <tr><td colSpan={10} className="empty">No vehicle journeys match the selected criteria</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Pagination Controls */}
            <div className="pagination-bar">
              <div style={{ fontSize: 11, color: '#5d7089', fontWeight: 600 }}>
                Showing {filteredJourneys.length > 0 ? (currentPage - 1) * pageSize + 1 : 0}–{Math.min(currentPage * pageSize, filteredJourneys.length)} of {filteredJourneys.length} journeys
                {filteredJourneys.length !== journeyData.length && (
                  <span style={{ color: '#8ca0bc', marginLeft: 4 }}>
                    (filtered from {journeyData.length} total)
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 10, color: '#8ca0bc', fontWeight: 700 }}>Rows:</span>
                  <select
                    value={pageSize}
                    onChange={e => setPageSize(Number(e.target.value))}
                    style={{ padding: '4px 8px', border: '1px solid #dfe6f0', borderRadius: 6, fontSize: 10, fontWeight: 700 }}
                  >
                    <option value={15}>15</option>
                    <option value={25}>25</option>
                    <option value={50}>50</option>
                    <option value={100}>100</option>
                  </select>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <button
                    className="page-btn"
                    disabled={currentPage <= 1}
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  >
                    ‹ Prev
                  </button>
                  <span style={{ fontSize: 11, fontWeight: 700, padding: '0 6px', color: '#37465e' }}>
                    {currentPage} / {totalPages}
                  </span>
                  <button
                    className="page-btn"
                    disabled={currentPage >= totalPages}
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  >
                    Next ›
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* ── Inflow Volume Timeline Card ── */}
          <InflowVolumeTimelineCard
            analytics={analytics}
            timeFilter={timeFilter}
            loading={jtLoading}
          />

          {/* ── Remaining Cameras — Live Only (Never Saved to DB) ── */}
          <OtherCamerasLiveCard remaining={vehicleCounts?.remaining_cameras_live} />
        </div>

        {/* Right: Engine Status + Rules + Registered Cameras */}
        <div className="span4" style={{ display: 'grid', gap: 16, alignContent: 'start' }}>
          {/* Engine Status */}
          <div className="card">
            <div className="head">
              <h3>⚙️ Engine Status</h3>
              <span className={`badge ${engine?.running ? 'on_site' : 'err'}`}>
                {engine?.running ? 'RUNNING' : 'OFFLINE'}
              </span>
            </div>
            <div className="body" style={{ fontSize: 11 }}>
              <div style={{ display: 'grid', gap: 6 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="muted">Model</span>
                  <b>{engine?.model || '—'}</b>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="muted">Total Scans</span>
                  <b>{engine?.scans ?? '—'}</b>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="muted">Total Sightings</span>
                  <b>{engine?.sightings ?? '—'}</b>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="muted">Passages Recorded</span>
                  <b>{engine?.passages ?? '—'}</b>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="muted">Object Backends</span>
                  <b>{engine?.objects_backends ?? '—'}</b>
                </div>
              </div>
            </div>
          </div>

          {/* Journey Rules */}
          <div className="card">
            <div className="head"><h3>🔧 Journey Rules</h3></div>
            <div className="body">
              <RulesPanel rules={rules} onUpdate={loadState} />
            </div>
          </div>

          {/* Registered Cameras */}
          <div className="card">
            <div className="head"><h3>📹 Cameras</h3></div>
            <div className="body">
              <RegisteredCameras
                devices={devices}
                activeSlots={activeSlots}
                onToggleViewSlot={handleToggleViewSlot}
                onEditCamera={d => setEditDevice(d)}
                onOpenAdd={() => setAddOpen(true)}
              />
            </div>
          </div>

          {/* Gate Vehicle Type Breakdown (Saved DB History + Remaining Cameras Live Only) */}
          <SecurityVehicleCountsCard
            data={vehicleCounts}
            range={vcRange}
            setRange={setVcRange}
            customStart={vcCustomStart}
            setCustomStart={setVcCustomStart}
            customEnd={vcCustomEnd}
            setCustomEnd={setVcCustomEnd}
            loading={vcLoading}
            onOpenHistory={(scope) => setHistoryScope(scope || 'main')}
          />
        </div>
      </div>

      {/* ── Bottom Full-Width Ambulance Registry & Email Dispatch ── */}
      <AmbulanceRegistryCard
        data={ambulanceData}
        rangeHours={ambHours}
        setRangeHours={setAmbHours}
        loading={ambLoading}
        onRefresh={() => fetchAmbulances(true)}
        onOpenSightingPhoto={(info) => setSelectedSightingModal(info)}
        onOpenDispatch={(payload) => setDispatchAmbulance(payload)}
      />

      {/* ── Lightbox & Modals ── */}
      <AmbulanceEmailDispatchModal
        alertData={dispatchAmbulance}
        onClose={() => setDispatchAmbulance(null)}
        onDispatched={() => loadState()}
      />
      <VehicleHistoryModal
        open={!!historyScope}
        scope={historyScope || 'main'}
        onClose={() => setHistoryScope(null)}
        initialData={vehicleCounts}
      />
      <FullscreenModal
        stream={fullscreenStream}
        device={deviceMap[fullscreenStream]}
        worker={engine?.workers?.[fullscreenStream]}
        feedSource={feedSource}
        onClose={() => setFullscreenStream(null)}
      />

      <SightingModal
        sighting={selectedSightingModal}
        deviceMap={deviceMap}
        onClose={() => setSelectedSightingModal(null)}
      />

      <AddCameraModal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onAdd={() => loadState()}
      />

      <EditCameraModal
        device={editDevice}
        open={!!editDevice}
        onClose={() => setEditDevice(null)}
        onSave={() => { setEditDevice(null); loadState(); }}
        onDelete={() => { setEditDevice(null); loadState(); }}
      />
    </div>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error('Vehicle Tracker UI Error:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, fontFamily: 'Inter, sans-serif', maxWidth: 640, margin: '60px auto', background: '#fff', borderRadius: 12, boxShadow: '0 4px 24px rgba(0,0,0,0.08)', border: '1px solid #fee2e2' }}>
          <h2 style={{ color: '#dc2626', margin: '0 0 10px 0', fontSize: 20 }}>Vehicle Tracker UI Encountered an Error</h2>
          <p style={{ color: '#475569', fontSize: 13, margin: '0 0 16px 0' }}>An unexpected error occurred during rendering:</p>
          <pre style={{ background: '#fef2f2', padding: 14, borderRadius: 8, overflowX: 'auto', fontSize: 12, color: '#991b1b', border: '1px solid #fecaca' }}>
            {this.state.error?.toString()}
          </pre>
          <button
            onClick={() => window.location.reload()}
            style={{ marginTop: 18, padding: '9px 18px', background: '#1d4ed8', color: '#fff', border: 'none', borderRadius: 7, cursor: 'pointer', fontWeight: 700, fontSize: 13 }}
          >
            🔄 Reload Dashboard
          </button>
        </div>
      );
    }
    return this.props?.children || null;
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
