import XLSX from 'xlsx';

const tableHeaders = [
  'S.No', 'License Plate', 'Status', 'Entry Camera', 'Entry Date & Time',
  'Exit Camera', 'Exit Date & Time', 'Dwell Duration', 'Cameras Visited',
  'Total Reads', 'Entry Vehicle Photo', 'Exit Vehicle Photo'
];

const dummyJourneys = [
  {
    plate: 'HR26DQ5551',
    status: 'on_site',
    first_at: Date.now() - 600000,
    dwell_ms: 600000,
    entry: { device: 'Main Gate Entry (PTZ)', photo_url: '/sightings/abc/photo' },
    exit: null,
    cameras: [{ name: 'Main Gate Entry (PTZ)' }],
    reads: 2
  },
  {
    plate: 'DL0CCR0408',
    status: 'exited',
    first_at: Date.now() - 3600000,
    dwell_ms: 1800000,
    entry: { device: 'Main Gate Entry (PTZ)', photo_url: '/sightings/xyz/photo' },
    exit: { device: 'Exit Gate (Primary)', photo_url: '/sightings/exit/photo', at: Date.now() - 1800000 },
    cameras: [{ name: 'Main Gate Entry (PTZ)' }, { name: 'Exit Gate (Primary)' }],
    reads: 4
  }
];

const apiOrigin = 'http://124.123.89.162:31016';
const dataRows = dummyJourneys.map((j, idx) => {
  const entryTime = new Date(j.first_at).toLocaleString('en-IN');
  const exitTime = j.exit?.at ? new Date(j.exit.at).toLocaleString('en-IN') : '—';
  const path = (j.cameras || []).map(c => c.name).join(' → ') || j.entry?.device || '—';
  const entryPhotoUrl = j.entry?.photo_url ? `${apiOrigin}${j.entry.photo_url}` : null;
  const exitPhotoUrl = j.exit?.photo_url ? `${apiOrigin}${j.exit.photo_url}` : null;

  return [
    idx + 1,
    j.plate,
    j.status === 'on_site' ? 'ON SITE' : 'EXITED',
    j.entry?.device || '—',
    entryTime,
    j.exit?.device || '—',
    exitTime,
    '10m 0s',
    path,
    j.reads || 1,
    entryPhotoUrl ? { t: 's', v: 'View Entry Photo', l: { Target: entryPhotoUrl, Tooltip: 'Click to open' } } : '—',
    exitPhotoUrl ? { t: 's', v: 'View Exit Photo', l: { Target: exitPhotoUrl, Tooltip: 'Click to open' } } : '—',
  ];
});

const wb = XLSX.utils.book_new();
const ws = XLSX.utils.aoa_to_sheet([tableHeaders, ...dataRows]);
ws['!cols'] = [
  { wch: 6 }, { wch: 16 }, { wch: 12 }, { wch: 26 }, { wch: 22 },
  { wch: 26 }, { wch: 22 }, { wch: 16 }, { wch: 36 }, { wch: 12 },
  { wch: 20 }, { wch: 20 }
];
XLSX.utils.book_append_sheet(wb, ws, 'Vehicle Journeys');
XLSX.writeFile(wb, 'd:\\test_clean.xlsx');
console.log('Generated test_clean.xlsx successfully!');
