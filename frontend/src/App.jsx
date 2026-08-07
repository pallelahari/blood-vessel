import { useMemo, useState, useRef, useEffect, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip as RTooltip, ResponsiveContainer,
  ScatterChart, Scatter, PieChart, Pie, Cell, Legend,
  Line, CartesianGrid, ReferenceLine, ComposedChart, ErrorBar
} from "recharts";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000/api";

const VIEWS = [
  { key: "original",         label: "Original",       desc: "Raw grayscale input image" },
  { key: "binary_mask",      label: "Binary Mask",    desc: "GeoJSON-derived vessel mask" },
  { key: "blob_labeled",     label: "Blob Labels",    desc: "Connected components colored + ID'd" },
  { key: "skeleton_overlay", label: "Skeleton",       desc: "Centerline skeleton (green) on mask" },
  { key: "compact_overlay",  label: "Morphology Map", desc: "Blue = elongated / Red = compact blobs" },
];

// Defined at module scope (not inside App) so it keeps a stable component
// identity across re-renders. Previously this lived inside App's render
// body, which meant every App re-render created a *new* FileUpload function
// and forced React to unmount/remount its <input> DOM node — including
// while an OS file-picker dialog was still open (e.g. during the progress
// poll's periodic re-renders). If that swap happened mid-selection, the
// browser's file-selected event landed on an already-discarded input and
// never reached React, so the box silently failed to update.
// Radial profiles can have hundreds of distance bins — drawing a discrete
// error-bar whisker at every single one is unreadable. This picks ~targetCount
// evenly spaced indices (always including the first and last) so error bars
// render as distinct, legible marks instead of a solid mass.
function sampleIndices(length, targetCount = 30) {
  if (length <= targetCount) return Array.from({ length }, (_, i) => i);
  const stride = (length - 1) / (targetCount - 1);
  const idx = new Set();
  for (let i = 0; i < targetCount; i++) idx.add(Math.round(i * stride));
  return Array.from(idx).sort((a, b) => a - b);
}

function FileUpload({ label, accept, hint, file, onChange }) {
  return (
    <label className={`upload-zone${file ? " has-file" : ""}`}>
      <input type="file" accept={accept} onChange={e => onChange(e.target.files?.[0] || null)} />
      <span className="upload-zone-title">
        <svg width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
        {file ? file.name : label}
      </span>
      <span className="upload-zone-hint">{file ? "click to change" : hint}</span>
    </label>
  );
}

const MORPH_COLORS = {
  compact:      "#3ecf8e",
  elongated:    "#4f8ef7",
  intermediate: "#f5a623",
  degenerate:   "#555b77",
};

const MORPHOLOGY_FIELDS = [
  { key: "blob_id",                                 label: "Blob ID" },
  { key: "centroid_x",                              label: "Centroid X (px)",              decimals: 1 },
  { key: "centroid_y",                              label: "Centroid Y (px)",              decimals: 1 },
  { key: "morphology_2d",                           label: "Morphology" },
  { key: "circularity_2d",                          label: "Circularity 2D",               decimals: 3 },
  { key: "eccentricity_2d",                         label: "Eccentricity 2D",              decimals: 3 },
  { key: "area_px",                                 label: "Area (px²)",                   decimals: 0 },
  { key: "n_segments",                              label: "Segments",                     decimals: 0 },
  { key: "total_skeleton_length_px",                label: "Total Skeleton Length (px)",   decimals: 1 },
  { key: "max_segment_length_px",                   label: "Max Segment Length (px)",      decimals: 1 },
  { key: "centerline_network_diameter_px",          label: "Network Diameter (px)",        decimals: 1 },
  { key: "centerline_network_diameter_unmerged_px", label: "Network Diam. Unmerged (px)",  decimals: 1 },
  { key: "centerline_network_diameter_merged_px",   label: "Network Diam. Merged (px)",    decimals: 1 },
  { key: "width_mean_length_weighted_px",           label: "Width Mean LW (px)",           decimals: 2 },
  { key: "width_mean_over_segments_px",             label: "Width Mean Segments (px)",     decimals: 2 },
  { key: "width_along_longest_path_px",             label: "Width Longest Path (px)",      decimals: 2 },
  { key: "width_along_longest_unmerged_path_px",    label: "Width Longest Unmerged (px)",  decimals: 2 },
  { key: "width_along_longest_merged_path_px",      label: "Width Longest Merged (px)",    decimals: 2 },
  { key: "length_clean",                            label: "Length Clean (px)",            decimals: 2 },
  { key: "width_clean",                             label: "Width Clean (px)",             decimals: 2 },
];

function fmtVal(val, decimals) {
  if (val === undefined || val === null || val === "") return "—";
  if (typeof val === "string") return val;
  const n = Number(val);
  if (isNaN(n)) return String(val);
  return decimals === 0 ? Math.round(n).toLocaleString() : n.toFixed(decimals);
}

function MorphBadge({ val }) {
  const cls = `metric-morphology morph-${(val || "degenerate").toLowerCase()}`;
  return <span className={cls}>{val || "—"}</span>;
}

// ── Dashboard helpers ────────────────────────────────────
function avg(arr) { return arr.length ? arr.reduce((a,b) => a+b, 0) / arr.length : 0; }
function med(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a,b)=>a-b);
  const m = Math.floor(s.length/2);
  return s.length%2 ? s[m] : (s[m-1]+s[m])/2;
}

function StatCard({ label, value, unit, color }) {
  return (
    <div className="dash-stat-card">
      <span className="dash-stat-label">{label}</span>
      <span className="dash-stat-value" style={color ? { color } : {}}>{value}</span>
      {unit && <span className="dash-stat-unit">{unit}</span>}
    </div>
  );
}

function ChartCard({ title, children }) {
  return (
    <div className="dash-chart-card">
      <div className="dash-chart-title">{title}</div>
      {children}
    </div>
  );
}

function Dashboard({ blobs, imageName }) {
  if (!blobs || blobs.length === 0) {
    return <div className="dash-empty">No blob data available. Run analysis first.</div>;
  }

  const areas        = blobs.map(b => b.area_px).filter(v => v != null);
  const circs        = blobs.map(b => b.circularity_2d).filter(v => v != null);
  const lengths      = blobs.map(b => b.length_clean).filter(v => v != null);
  const widths       = blobs.map(b => b.width_clean).filter(v => v != null);
  const skelLengths  = blobs.map(b => b.total_skeleton_length_px).filter(v => v != null);
  const segments     = blobs.map(b => b.n_segments).filter(v => v != null);

  // Morphology breakdown for pie
  const morphCounts = {};
  blobs.forEach(b => {
    const m = b.morphology_2d || "degenerate";
    morphCounts[m] = (morphCounts[m] || 0) + 1;
  });
  const pieData = Object.entries(morphCounts).map(([name, value]) => ({ name, value }));

  // Area histogram — 10 bins
  const maxArea = Math.max(...areas);
  const binCount = 10;
  const binSize = maxArea / binCount || 1;
  const areaBins = Array.from({ length: binCount }, (_, i) => ({
    range: `${Math.round(i*binSize)}–${Math.round((i+1)*binSize)}`,
    count: areas.filter(a => a >= i*binSize && a < (i+1)*binSize).length,
  }));

  // Circularity histogram — 10 bins 0..1
  const circBins = Array.from({ length: 10 }, (_, i) => ({
    range: `${(i*0.1).toFixed(1)}–${((i+1)*0.1).toFixed(1)}`,
    count: circs.filter(c => c >= i*0.1 && c < (i+1)*0.1).length,
  }));

  // Scatter: length vs width (top 80 blobs max for perf)
  const scatterData = blobs.slice(0, 80).map(b => ({
    x: b.length_clean ?? 0,
    y: b.width_clean ?? 0,
    fill: MORPH_COLORS[b.morphology_2d] || MORPH_COLORS.degenerate,
  }));

  // Skeleton length bar — top 20 blobs by skeleton length
  const topSkel = [...blobs]
    .filter(b => b.total_skeleton_length_px != null)
    .sort((a,b) => b.total_skeleton_length_px - a.total_skeleton_length_px)
    .slice(0, 20)
    .map(b => ({ name: `#${b.blob_id}`, value: Math.round(b.total_skeleton_length_px) }));

  return (
    <div className="dashboard">
      {/* Header */}
      <div className="dash-header">
        <div>
          <div className="dash-header-title">Analysis Dashboard</div>
          {imageName && <div className="dash-header-sub">{imageName}</div>}
        </div>
        <div className="dash-header-count">{blobs.length} blobs analysed</div>
      </div>

      {/* Summary stat cards */}
      <div className="dash-stats-grid">
        <StatCard label="Total Blobs"       value={blobs.length}                                          />
        <StatCard label="Avg Area"          value={Math.round(avg(areas)).toLocaleString()} unit="px²"    />
        <StatCard label="Median Area"       value={Math.round(med(areas)).toLocaleString()} unit="px²"    />
        <StatCard label="Avg Circularity"   value={avg(circs).toFixed(3)}                                 />
        <StatCard label="Avg Length"        value={avg(lengths).toFixed(1)}  unit="px"                    />
        <StatCard label="Avg Width"         value={avg(widths).toFixed(1)}   unit="px"                    />
        <StatCard label="Avg Skeleton Len"  value={avg(skelLengths).toFixed(1)} unit="px"                 />
        <StatCard label="Avg Segments"      value={avg(segments).toFixed(1)}                              />
        <StatCard label="Compact"           value={morphCounts.compact || 0}      color={MORPH_COLORS.compact}      />
        <StatCard label="Elongated"         value={morphCounts.elongated || 0}    color={MORPH_COLORS.elongated}    />
        <StatCard label="Intermediate"      value={morphCounts.intermediate || 0} color={MORPH_COLORS.intermediate} />
        <StatCard label="Degenerate"        value={morphCounts.degenerate || 0}   color={MORPH_COLORS.degenerate}   />
      </div>

      {/* Charts row 1 */}
      <div className="dash-charts-row">
        <ChartCard title="Morphology Breakdown">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({name,percent}) => `${name} ${(percent*100).toFixed(0)}%`} labelLine={false}>
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={MORPH_COLORS[entry.name] || "#555b77"} />
                ))}
              </Pie>
              <Legend formatter={v => <span style={{color:"#8b90a8",fontSize:12}}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Area Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={areaBins} margin={{ top:4, right:8, left:0, bottom:40 }}>
              <XAxis dataKey="range" tick={{ fill:"#555b77", fontSize:9 }} angle={-35} textAnchor="end" interval={0} />
              <YAxis tick={{ fill:"#555b77", fontSize:11 }} />
              <RTooltip contentStyle={{ background:"#13161e", border:"1px solid #222738", color:"#e8eaf2", fontSize:12 }} />
              <Bar dataKey="count" fill="#4f8ef7" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Circularity Distribution">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={circBins} margin={{ top:4, right:8, left:0, bottom:40 }}>
              <XAxis dataKey="range" tick={{ fill:"#555b77", fontSize:10 }} angle={-35} textAnchor="end" interval={0} />
              <YAxis tick={{ fill:"#555b77", fontSize:11 }} />
              <RTooltip contentStyle={{ background:"#13161e", border:"1px solid #222738", color:"#e8eaf2", fontSize:12 }} />
              <Bar dataKey="count" fill="#3ecf8e" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Charts row 2 */}
      <div className="dash-charts-row">
        <ChartCard title="Length vs Width (per blob, colored by morphology)">
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart margin={{ top:4, right:16, left:0, bottom:4 }}>
              <XAxis dataKey="x" name="Length" unit="px" tick={{ fill:"#555b77", fontSize:11 }} label={{ value:"Length (px)", position:"insideBottom", offset:-4, fill:"#555b77", fontSize:11 }} />
              <YAxis dataKey="y" name="Width" unit="px"  tick={{ fill:"#555b77", fontSize:11 }} label={{ value:"Width (px)", angle:-90, position:"insideLeft", fill:"#555b77", fontSize:11 }} />
              <RTooltip cursor={{ fill:"rgba(255,255,255,0.04)" }} contentStyle={{ background:"#13161e", border:"1px solid #222738", color:"#e8eaf2", fontSize:12 }}
                formatter={(val, name) => [`${Number(val).toFixed(1)} px`, name]} />
              <Scatter data={scatterData} fill="#4f8ef7">
                {scatterData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Top 20 Blobs by Skeleton Length">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={topSkel} layout="vertical" margin={{ top:4, right:16, left:24, bottom:4 }}>
              <XAxis type="number" tick={{ fill:"#555b77", fontSize:11 }} unit="px" />
              <YAxis type="category" dataKey="name" tick={{ fill:"#8b90a8", fontSize:10 }} width={36} />
              <RTooltip contentStyle={{ background:"#13161e", border:"1px solid #222738", color:"#e8eaf2", fontSize:12 }} formatter={v => [`${v} px`, "Skeleton Length"]} />
              <Bar dataKey="value" fill="#7c5cfc" radius={[0,3,3,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────
export default function App() {
  const [imageFile, setImageFile]     = useState(null);
  const [geoFile, setGeoFile]         = useState(null);
  const [result, setResult]           = useState(null);
  const [status, setStatus]           = useState("Ready — upload an image and GeoJSON to begin.");
  const [statusType, setStatusType]   = useState("idle");
  const [activePage, setActivePage]   = useState("viewer");   // "viewer" | "dashboard"
  const [viewIndex, setViewIndex]     = useState(0);
  const [branchMode, setBranchMode]   = useState(false);
  const [branchPoints, setBranchPoints] = useState([]);
  const [redoStack, setRedoStack]     = useState([]);
  const [tooltip, setTooltip]         = useState(null);
  const [hoveredBlob, setHoveredBlob] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [progressLogs, setProgressLogs] = useState([]);
  const [toast, setToast]             = useState(null);
  const imgRef   = useRef(null);
  const pollRef  = useRef(null);
  const toastRef = useRef(null);
  const canvasRef = useRef(null);

  // Crop selection state
  const [cropMode, setCropMode]       = useState(false);
  const [cropRect, setCropRect]       = useState(null);   // { x1,y1,x2,y2 } in image px
  const [drawing, setDrawing]         = useState(false);
  const [drawStart, setDrawStart]     = useState(null);   // screen px
  const [drawCurrent, setDrawCurrent] = useState(null);   // screen px
  const [cropResult, setCropResult]   = useState(null);   // result for the cropped region
  const [cropLoading, setCropLoading] = useState(false);

  // ── Cross-correlation (standalone, independent of the main analysis) ──
  const [xcorrFileA, setXcorrFileA]   = useState(null);
  const [xcorrFileB, setXcorrFileB]   = useState(null);
  const [xcorrResult, setXcorrResult] = useState(null);
  const [xcorrLoading, setXcorrLoading] = useState(false);
  const [xcorrError, setXcorrError]   = useState("");

  // X-axis (distance) range filters for the two radial-profile charts —
  // full radial profiles run out to hundreds of px, but the informative
  // part is usually the first small fraction near the peak.
  const [acfXMin, setAcfXMin] = useState(0);
  const [acfXMax, setAcfXMax] = useState(null);   // null = full range
  const [xcorrXMin, setXcorrXMin] = useState(0);
  const [xcorrXMax, setXcorrXMax] = useState(null);

  const setMsg = (msg, type = "idle") => { setStatus(msg); setStatusType(type); };

  function showToast(msg, type = "success") {
    if (toastRef.current) clearTimeout(toastRef.current);
    setToast({ msg, type });
    toastRef.current = setTimeout(() => setToast(null), 4000);
  }

  const currentSrc = useMemo(() => {
    if (!result) return "";
    return result.images[VIEWS[viewIndex].key];
  }, [result, viewIndex]);

  async function processFiles() {
    if (!imageFile || !geoFile) { setMsg("Select both image and GeoJSON files.", "error"); return; }
    const fd = new FormData();
    fd.append("image_file", imageFile);
    fd.append("geojson_file", geoFile);
    setMsg("Processing… this may take a few seconds.", "processing");
    setLoading(true);
    setProgressLogs([]);

    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/progress`);
        const d = await r.json();
        if (d.logs) setProgressLogs([...d.logs]);
      } catch (_) {}
    }, 1000);

    try {
      const res  = await fetch(`${API_BASE}/process`, { method: "POST", body: fd });
      const body = await res.json();
      clearInterval(pollRef.current);
      if (!res.ok) throw new Error(body.error || "Processing failed");
      setResult(body);
      setBranchPoints([]); setRedoStack([]); setViewIndex(0);
      setProgressLogs([]);
      setActivePage("viewer");
      setMsg(`Done — output saved to: ${body.output_dir}`, "done");
      showToast("✓ Analysis complete! Switch to Dashboard to see charts.", "success");
    } catch (err) {
      clearInterval(pollRef.current);
      setMsg(`Error: ${err.message || err}`, "error");
    } finally { setLoading(false); }
  }

  async function onMouseMove(evt) {
    if (!result || !imgRef.current || activePage !== "viewer") return;
    const rect = imgRef.current.getBoundingClientRect();
    const x = Math.floor((evt.clientX - rect.left) * (result.width / rect.width));
    const y = Math.floor((evt.clientY - rect.top)  * (result.height / rect.height));
    try {
      const res  = await fetch(`${API_BASE}/blob-info?x=${x}&y=${y}`);
      const body = await res.json();
      if (!res.ok || !body.blob_id) { setTooltip(null); setHoveredBlob(null); return; }
      setHoveredBlob(body);
      setTooltip({ left: evt.clientX + 14, top: evt.clientY + 10, blob: body });
    } catch (_) { setTooltip(null); setHoveredBlob(null); }
  }

  function onImageClick(evt) {
    if (!result || !branchMode || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const x = Math.floor((evt.clientX - rect.left) * (result.width / rect.width));
    const y = Math.floor((evt.clientY - rect.top)  * (result.height / rect.height));
    const next = [...branchPoints, { x, y }];
    setBranchPoints(next); setRedoStack([]);
    setMsg(`Branch point added at (${x}, ${y}) — total: ${next.length}`, "done");
  }

  function undo()     { if (!branchPoints.length) return; setRedoStack([...redoStack, branchPoints[branchPoints.length-1]]); setBranchPoints(branchPoints.slice(0,-1)); }
  function redo()     { if (!redoStack.length) return; setBranchPoints([...branchPoints, redoStack[redoStack.length-1]]); setRedoStack(redoStack.slice(0,-1)); }
  function clearPts() { setBranchPoints([]); setRedoStack([]); }

  async function saveBranching() {
    if (!result) return;
    setMsg("Saving branching data…", "processing");
    try {
      const res  = await fetch(`${API_BASE}/finalize-branches`, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({ points: branchPoints }) });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || "Save failed");
      setMsg(`Saved — CSV: ${body.csv_path}`, "done");
      showToast(`✓ Saved! ${body.csv_path.split("\\").pop().split("/").pop()}`, "success");
    } catch (err) {
      setMsg(`Error: ${err.message || err}`, "error");
      showToast(`✕ Save failed: ${err.message || err}`, "error");
    }
  }

  // ── Crop helpers ────────────────────────────────────────
  function toImgCoords(screenX, screenY) {
    if (!imgRef.current) return { x: 0, y: 0 };
    const rect = imgRef.current.getBoundingClientRect();
    return {
      x: Math.round(Math.max(0, Math.min(result.width,  (screenX - rect.left) * (result.width  / rect.width)))),
      y: Math.round(Math.max(0, Math.min(result.height, (screenY - rect.top)  * (result.height / rect.height)))),
    };
  }

  function onCropMouseDown(e) {
    if (!cropMode || !result) return;
    e.preventDefault();
    setDrawing(true);
    setDrawStart({ x: e.clientX, y: e.clientY });
    setDrawCurrent({ x: e.clientX, y: e.clientY });
    setCropRect(null);
  }

  function onCropMouseMove(e) {
    if (!drawing) return;
    setDrawCurrent({ x: e.clientX, y: e.clientY });
    // also block hover tooltip while drawing
    setTooltip(null); setHoveredBlob(null);
  }

  function onCropMouseUp(e) {
    if (!drawing || !drawStart) return;
    setDrawing(false);
    const p1 = toImgCoords(drawStart.x, drawStart.y);
    const p2 = toImgCoords(e.clientX, e.clientY);
    const x1 = Math.min(p1.x, p2.x), x2 = Math.max(p1.x, p2.x);
    const y1 = Math.min(p1.y, p2.y), y2 = Math.max(p1.y, p2.y);
    if (x2 - x1 < 10 || y2 - y1 < 10) { setCropRect(null); return; } // too small
    setCropRect({ x1, y1, x2, y2 });
    setDrawStart(null); setDrawCurrent(null);
  }

  // Screen rect of the drawn selection (for rendering the dashed box)
  function screenRect() {
    if (!imgRef.current) return null;
    const imgRect = imgRef.current.getBoundingClientRect();
    const scaleX  = imgRect.width  / result.width;
    const scaleY  = imgRect.height / result.height;
    const s = drawStart, c = drawCurrent;
    if (!s || !c) return null;
    const left   = Math.min(s.x, c.x) - imgRect.left;
    const top    = Math.min(s.y, c.y) - imgRect.top;
    const width  = Math.abs(c.x - s.x);
    const height = Math.abs(c.y - s.y);
    return { left, top, width, height };
  }

  // Confirmed crop rect in screen coords (for showing after mouse up)
  function confirmedScreenRect() {
    if (!cropRect || !imgRef.current || !result) return null;
    const imgRect = imgRef.current.getBoundingClientRect();
    const scaleX  = imgRect.width  / result.width;
    const scaleY  = imgRect.height / result.height;
    return {
      left:   cropRect.x1 * scaleX,
      top:    cropRect.y1 * scaleY,
      width:  (cropRect.x2 - cropRect.x1) * scaleX,
      height: (cropRect.y2 - cropRect.y1) * scaleY,
    };
  }

  async function runCropAnalysis() {
    if (!cropRect || !imageFile || !geoFile) return;
    setCropLoading(true);
    setCropResult(null);
    setMsg("Running crop analysis…", "processing");
    const fd = new FormData();
    fd.append("image_file", imageFile);
    fd.append("geojson_file", geoFile);
    fd.append("x1", cropRect.x1);
    fd.append("y1", cropRect.y1);
    fd.append("x2", cropRect.x2);
    fd.append("y2", cropRect.y2);
    fd.append("image_name", imageFile.name);
    try {
      const res  = await fetch(`${API_BASE}/crop-process`, { method: "POST", body: fd });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || "Crop analysis failed");
      setCropResult(body);
      setMsg(`Crop done — ${body.blobs?.length ?? 0} blobs in selected region`, "done");
      showToast(`✓ Crop analysis done! ${body.blobs?.length ?? 0} blobs found`, "success");
    } catch (err) {
      setMsg(`Error: ${err.message || err}`, "error");
      showToast(`✕ ${err.message || err}`, "error");
    } finally { setCropLoading(false); }
  }

  function clearCrop() {
    setCropRect(null); setCropResult(null);
    setDrawStart(null); setDrawCurrent(null);
  }

  async function runCrossCorrelation() {
    if (!xcorrFileA || !xcorrFileB) return;
    setXcorrLoading(true);
    setXcorrError("");
    setXcorrResult(null);
    const fd = new FormData();
    fd.append("file_a", xcorrFileA);
    fd.append("file_b", xcorrFileB);
    try {
      const res  = await fetch(`${API_BASE}/cross-correlate`, { method: "POST", body: fd });
      const body = await res.json();
      if (!res.ok) throw new Error(body.error || "Cross-correlation failed");
      setXcorrResult(body);
    } catch (err) {
      setXcorrError(err.message || String(err));
    } finally { setXcorrLoading(false); }
  }

  // Auto-zoom the radial-profile x-axis to a sensible default range whenever
  // a new result comes in (full domain runs to hundreds of px; the decay
  // near the peak is what's usually informative). The slider in the chart
  // still lets you widen it back out to the full range.
  useEffect(() => {
    const rad = result?.acf_radial?.radius;
    if (rad && rad.length) {
      const full = rad[rad.length - 1];
      const lc = result.acf_radial.correlation_length || 0;
      const rh = result.acf_radial.half_max_distance || 0;
      const suggested = Math.max(lc, rh, 10) * 4;
      setAcfXMin(0);
      setAcfXMax(Math.min(full, suggested));
    }
  }, [result]);

  useEffect(() => {
    const rad = xcorrResult?.radius;
    if (rad && rad.length) {
      const full = rad[rad.length - 1];
      const rh = xcorrResult.half_max_distance || 0;
      const suggested = Math.max(rh, 10) * 4;
      setXcorrXMin(0);
      setXcorrXMax(Math.min(full, suggested));
    }
  }, [xcorrResult]);

  return (
    <div className="app">
      {/* TOPBAR */}
      <header className="topbar">
        <div className="topbar-logo"><span className="dot" />VesselSeg</div>

        {/* PAGE TABS */}
        <div className="topbar-tabs">
          <button className={`tab-btn${activePage === "viewer" ? " active" : ""}`} onClick={() => setActivePage("viewer")}>
            🔬 Viewer
          </button>
          <button className={`tab-btn${activePage === "dashboard" ? " active" : ""}`} onClick={() => setActivePage("dashboard")} disabled={!result}>
            📊 Dashboard {result ? `(${result.blobs?.length ?? 0} blobs)` : ""}
          </button>
          <button className={`tab-btn${activePage === "correlation" ? " active" : ""}`} onClick={() => setActivePage("correlation")}>
            🔗 Correlation
          </button>
        </div>

        <div className={`topbar-status ${statusType}`}>{status}</div>
      </header>

      {/* SIDEBAR — always visible */}
      <aside className="sidebar">
        <div className="sidebar-section">
          <span className="sidebar-label">Input Files</span>
          <FileUpload label="Upload Image" accept="image/*" hint=".png / .jpg / .tif" file={imageFile} onChange={setImageFile} />
          <FileUpload label="Upload GeoJSON" accept=".geojson,.json" hint=".geojson / .json" file={geoFile} onChange={setGeoFile} />
          <button className={`btn-process${loading ? " loading" : ""}`} onClick={processFiles} disabled={loading || !imageFile || !geoFile}>
            {loading ? "⏳  Processing…" : "▶  Run Analysis"}
          </button>
        </div>

        {activePage === "viewer" && <>
          <div className="sidebar-section">
            <span className="sidebar-label">View</span>
            <div className="view-grid">
              {VIEWS.map((v, i) => (
                <button key={v.key} className={`view-btn${viewIndex === i ? " active" : ""}`} onClick={() => setViewIndex(i)} title={v.desc}>
                  {v.label}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-section">
            <span className="sidebar-label">Branching Annotation</span>
            <div className="branch-controls">
              <button className={`branch-toggle${branchMode ? " active" : ""}`} onClick={() => setBranchMode(!branchMode)}>
                {branchMode ? "🔴  Marking ON — click image" : "⚪  Enable Branch Marking"}
              </button>
              <div className="branch-actions">
                <button onClick={undo} disabled={!branchPoints.length}>↩ Undo</button>
                <button onClick={redo} disabled={!redoStack.length}>↪ Redo</button>
                <button onClick={clearPts} disabled={!branchPoints.length && !redoStack.length}>✕ Clear</button>
              </div>
              <div className="branch-count">{branchPoints.length} point{branchPoints.length !== 1 ? "s" : ""} marked</div>
              <button className="btn-save" onClick={saveBranching} disabled={!result || !branchPoints.length}>
                ✓  Save Branching + Recompile
              </button>
            </div>
          </div>

          <div className="sidebar-section">
            <span className="sidebar-label">Region Analysis</span>
            <div style={{fontSize:11, color:"var(--text3)", padding:"0 4px 6px"}}>
              Draw a rectangle on the image to analyse only that region.
            </div>
            <button
              className={`branch-toggle${cropMode ? " active" : ""}`}
              style={cropMode ? {borderColor:"rgba(124,92,252,0.5)", color:"#7c5cfc", background:"rgba(124,92,252,0.1)"} : {}}
              onClick={() => { setCropMode(!cropMode); clearCrop(); setBranchMode(false); }}>
              {cropMode ? "✏️  Drawing ON — drag image" : "⬚  Enable Region Select"}
            </button>
            {cropRect && (
              <div style={{fontSize:11, fontFamily:"var(--mono)", color:"var(--text2)", background:"var(--bg3)", border:"1px solid var(--border2)", borderRadius:6, padding:"6px 10px", lineHeight:1.8}}>
                <div>x1={cropRect.x1}  y1={cropRect.y1}</div>
                <div>x2={cropRect.x2}  y2={cropRect.y2}</div>
                <div style={{color:"var(--text3)"}}>Size: {cropRect.x2-cropRect.x1} × {cropRect.y2-cropRect.y1} px</div>
              </div>
            )}
            <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:5}}>
              <button className="branch-actions" style={{padding:"7px 6px", background:"var(--bg3)", border:"1px solid var(--border)", borderRadius:6, fontSize:11, color:"var(--text2)", cursor:"pointer"}}
                onClick={clearCrop} disabled={!cropRect}>
                ✕ Clear
              </button>
              <button style={{padding:"7px 6px", background: cropRect ? "rgba(124,92,252,0.15)" : "var(--bg3)", border:`1px solid ${cropRect ? "rgba(124,92,252,0.4)" : "var(--border)"}`, borderRadius:6, fontSize:11, color: cropRect ? "#7c5cfc" : "var(--text3)", cursor: cropRect ? "pointer" : "default", fontFamily:"var(--sans)"}}
                onClick={runCropAnalysis} disabled={!cropRect || cropLoading}>
                {cropLoading ? "⏳…" : "▶ Analyse"}
              </button>
            </div>
          </div>

          <div className="sidebar-section">
            <span className="sidebar-label">Current View</span>
            {result
              ? <div style={{ fontSize:11, color:"var(--text3)", lineHeight:1.6, padding:"0 4px" }}>
                  <div style={{ color:"var(--text2)", fontWeight:500, marginBottom:2 }}>{VIEWS[viewIndex].label}</div>
                  {VIEWS[viewIndex].desc}
                </div>
              : <div style={{ fontSize:11, color:"var(--text3)", padding:"0 4px" }}>Run analysis to view results</div>
            }
          </div>
        </>}

        {activePage === "dashboard" && result && (
          <div className="sidebar-section">
            <span className="sidebar-label">Run Info</span>
            <div style={{ fontSize:11, color:"var(--text3)", lineHeight:1.8, padding:"0 4px" }}>
              <div><span style={{color:"var(--text2)"}}>Image:</span> {imageFile?.name || "—"}</div>
              <div><span style={{color:"var(--text2)"}}>Size:</span> {result.width} × {result.height} px</div>
              <div><span style={{color:"var(--text2)"}}>Blobs:</span> {result.blobs?.length ?? 0}</div>
              <div style={{marginTop:8, wordBreak:"break-all"}}><span style={{color:"var(--text2)"}}>Output:</span> {result.output_dir?.split("\\").pop() || result.output_dir}</div>
              {result.reports?.autocorrelation_html && (
                <div style={{marginTop:10}}>
                  <a
                    href={`${API_BASE}/output-file/${result.reports.autocorrelation_html}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      display:"inline-block", fontSize:11, color:"#fff",
                      background:"var(--accent, #4f8cff)", padding:"6px 10px",
                      borderRadius:6, textDecoration:"none", fontWeight:500,
                    }}
                  >
                    View 2D Autocorrelation Report ↗
                  </a>
                </div>
              )}
            </div>
          </div>
        )}
      </aside>

      {/* MAIN */}
      <main className="main">
        {/* ── VIEWER PAGE ── */}
        {activePage === "viewer" && <>
          {result && (
            <div className="stats-strip">
              <div className="stat-cell"><span className="stat-cell-label">Width</span><span className="stat-cell-value">{result.width}</span><span className="stat-cell-unit">px</span></div>
              <div className="stat-cell"><span className="stat-cell-label">Height</span><span className="stat-cell-value">{result.height}</span><span className="stat-cell-unit">px</span></div>
              {hoveredBlob && hoveredBlob.blob_id > 0 && <>
                <div className="stat-cell"><span className="stat-cell-label">Hovered Blob</span><span className="stat-cell-value">#{hoveredBlob.blob_id}</span></div>
                <div className="stat-cell"><span className="stat-cell-label">Morphology</span><span className="stat-cell-value" style={{fontSize:13}}>{hoveredBlob.morphology_2d || "—"}</span></div>
                <div className="stat-cell"><span className="stat-cell-label">Circularity</span><span className="stat-cell-value">{hoveredBlob.circularity_2d != null ? Number(hoveredBlob.circularity_2d).toFixed(3) : "—"}</span></div>
                <div className="stat-cell"><span className="stat-cell-label">Area</span><span className="stat-cell-value">{hoveredBlob.area_px != null ? Math.round(hoveredBlob.area_px).toLocaleString() : "—"}</span><span className="stat-cell-unit">px²</span></div>
              </>}
            </div>
          )}

          <div className="viewer-area">
            {!result ? (
              <div className="viewer-empty">
                <div className="viewer-empty-icon">{loading ? "⚙️" : "🔬"}</div>
                {loading ? (
                  <div style={{width:"100%",maxWidth:500}}>
                    <p style={{marginBottom:12,color:"var(--amber)"}}>Analyzing vessels — please wait…</p>
                    <div style={{background:"var(--bg3)",border:"1px solid var(--border2)",borderRadius:"var(--radius)",padding:"12px 16px",fontFamily:"var(--mono)",fontSize:12,lineHeight:2,maxHeight:260,overflowY:"auto"}}>
                      {progressLogs.length === 0 && <span style={{color:"var(--text3)"}}>Starting up…</span>}
                      {progressLogs.map((l,i) => (
                        <div key={i} style={{color: i===progressLogs.length-1 ? "var(--green)" : "var(--text2)"}}>
                          <span style={{color:"var(--text3)",marginRight:8}}>›</span>{l}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <p>Upload files and run analysis to see vessel images</p>
                )}
              </div>
            ) : (
              <div className="viewer-wrap"
                onMouseDown={cropMode ? onCropMouseDown : undefined}
                onMouseMove={cropMode ? onCropMouseMove : onMouseMove}
                onMouseUp={cropMode ? onCropMouseUp : undefined}
                onMouseLeave={() => { setTooltip(null); setHoveredBlob(null); if (drawing) { setDrawing(false); } }}
                style={{ cursor: cropMode ? "crosshair" : undefined }}>
                <img ref={imgRef} src={currentSrc} alt="vessel view" className="viewer-img"
                  onMouseMove={!cropMode ? onMouseMove : undefined}
                  onMouseLeave={!cropMode ? () => { setTooltip(null); setHoveredBlob(null); } : undefined}
                  onClick={!cropMode ? onImageClick : undefined}
                  draggable={false}
                  style={{ userSelect:"none", pointerEvents: cropMode ? "none" : undefined }} />

                {/* Branch markers */}
                <div className="marker-layer">
                  {!cropMode && branchPoints.map((p,idx) => (
                    <div key={`${p.x}-${p.y}-${idx}`} className="marker"
                      style={{ left:`${(p.x/result.width)*100}%`, top:`${(p.y/result.height)*100}%` }} />
                  ))}

                  {/* Live draw rect while dragging */}
                  {drawing && drawStart && drawCurrent && (() => {
                    const r = screenRect();
                    return r ? (
                      <div style={{ position:"absolute", left:r.left, top:r.top, width:r.width, height:r.height,
                        border:"2px dashed #7c5cfc", background:"rgba(124,92,252,0.1)", pointerEvents:"none" }} />
                    ) : null;
                  })()}

                  {/* Confirmed selection */}
                  {cropRect && !drawing && (() => {
                    const r = confirmedScreenRect();
                    return r ? (
                      <div style={{ position:"absolute", left:r.left, top:r.top, width:r.width, height:r.height,
                        border:"2px solid #7c5cfc", background:"rgba(124,92,252,0.08)", pointerEvents:"none", boxShadow:"0 0 0 9999px rgba(0,0,0,0.35)" }}>
                        <div style={{ position:"absolute", top:-20, left:0, fontSize:10, color:"#7c5cfc", background:"rgba(13,15,20,0.9)", padding:"2px 6px", borderRadius:4, fontFamily:"var(--mono)", whiteSpace:"nowrap" }}>
                          {cropRect.x2-cropRect.x1} × {cropRect.y2-cropRect.y1} px
                        </div>
                      </div>
                    ) : null;
                  })()}
                </div>
              </div>
            )}
          </div>

          {hoveredBlob && hoveredBlob.blob_id > 0 && (
            <div className="metrics-panel">
              <div className="metrics-title">Blob #{hoveredBlob.blob_id} — All Metrics</div>
              <div className="metrics-grid">
                {MORPHOLOGY_FIELDS.map(f => (
                  <div className="metric-card" key={f.key}>
                    <span className="metric-card-label">{f.label}</span>
                    {f.key === "morphology_2d"
                      ? <MorphBadge val={hoveredBlob[f.key]} />
                      : <span className={`metric-card-value${f.key==="blob_id"?" highlight":""}`}>{fmtVal(hoveredBlob[f.key], f.decimals)}</span>
                    }
                  </div>
                ))}
              </div>
            </div>
          )}
        </>}

        {/* ── CROP RESULT PANEL ── */}
        {activePage === "viewer" && cropResult && (() => {
          const blobs   = cropResult.blobs || [];
          const avgArr  = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
          const areas   = blobs.map(b=>b.area_px).filter(Boolean);
          const circs   = blobs.map(b=>b.circularity_2d).filter(v=>v!=null);
          const lengths = blobs.map(b=>b.length_clean).filter(v=>v!=null);
          const widths  = blobs.map(b=>b.width_clean).filter(v=>v!=null);
          const morphCounts = {};
          blobs.forEach(b=>{ const m=b.morphology_2d||"degenerate"; morphCounts[m]=(morphCounts[m]||0)+1; });
          const folderName  = cropResult.output_dir?.split("\\").pop().split("/").pop() || cropResult.output_dir;
          const regionLabel = `${cropRect?.x2-cropRect?.x1} × ${cropRect?.y2-cropRect?.y1} px`;

          return (
            <div className="metrics-panel crop-result-panel">

              {/* ── Header ── */}
              <div style={{display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:16}}>
                <div>
                  <div className="metrics-title" style={{marginBottom:4}}>
                    ⬚ Region Analysis — {regionLabel} — {blobs.length} blobs
                  </div>
                  {/* Saved confirmation */}
                  <div style={{display:"flex", alignItems:"center", gap:6, fontSize:11, color:"var(--green)"}}>
                    <span>✓ Saved to analysis_outputs</span>
                    <span style={{color:"var(--text3)"}}>›</span>
                    <span style={{fontFamily:"var(--mono)", color:"var(--text2)", wordBreak:"break-all"}}>{folderName}</span>
                  </div>
                  <div style={{fontSize:11, color:"var(--text3)", marginTop:4, fontFamily:"var(--mono)"}}>
                    Contains: original.png · binary_mask.png · blob_labeled.png · skeleton_overlay.png · compact_overlay.png · blob_metrics.csv
                  </div>
                </div>
                <button onClick={()=>setCropResult(null)} style={{background:"none",border:"none",color:"var(--text3)",cursor:"pointer",fontSize:20,flexShrink:0,padding:"0 4px"}}>×</button>
              </div>

              {/* ── 5 overlay images ── */}
              <div style={{display:"flex", gap:10, overflowX:"auto", paddingBottom:10, marginBottom:16}}>
                {["original","binary_mask","blob_labeled","skeleton_overlay","compact_overlay"].map(key => (
                  cropResult.images?.[key] && (
                    <div key={key} style={{flexShrink:0, textAlign:"center"}}>
                      <div style={{fontSize:10, color:"var(--text3)", marginBottom:5, textTransform:"uppercase", letterSpacing:"0.7px"}}>
                        {key.replace(/_/g," ")}
                      </div>
                      <img src={cropResult.images[key]} alt={key}
                        style={{height:150, borderRadius:8, border:"1px solid var(--border2)", display:"block", boxShadow:"0 2px 8px rgba(0,0,0,0.3)"}} />
                    </div>
                  )
                ))}
              </div>

              {/* ── Summary stat cards ── */}
              <div style={{display:"grid", gridTemplateColumns:"repeat(auto-fill, minmax(130px,1fr))", gap:8, marginBottom:16}}>
                {[
                  ["Total Blobs",      blobs.length,                           "",    "var(--accent)"],
                  ["Avg Area",         Math.round(avgArr(areas)).toLocaleString(),"px²", null],
                  ["Avg Circularity",  avgArr(circs).toFixed(3),               "",    null],
                  ["Avg Length",       avgArr(lengths).toFixed(1),             "px",  null],
                  ["Avg Width",        avgArr(widths).toFixed(1),              "px",  null],
                  ["Compact",          morphCounts.compact||0,                 "",    "#3ecf8e"],
                  ["Elongated",        morphCounts.elongated||0,               "",    "#4f8ef7"],
                  ["Intermediate",     morphCounts.intermediate||0,            "",    "#f5a623"],
                  ["Degenerate",       morphCounts.degenerate||0,              "",    "#555b77"],
                ].map(([label,val,unit,color])=>(
                  <div key={label} className="metric-card">
                    <span className="metric-card-label">{label}</span>
                    <span className="metric-card-value" style={color?{color}:{}}>{val}</span>
                    {unit && <span style={{fontSize:10,color:"var(--text3)"}}>{unit}</span>}
                  </div>
                ))}
              </div>

              {/* ── Morphology bar chart ── */}
              {blobs.length > 0 && (
                <div>
                  <div style={{fontSize:11, fontWeight:600, color:"var(--text2)", textTransform:"uppercase", letterSpacing:"0.7px", marginBottom:10}}>
                    Morphology Breakdown
                  </div>
                  <div style={{display:"flex", gap:6, alignItems:"flex-end", height:80}}>
                    {Object.entries(morphCounts).map(([name, count])=>{
                      const pct = Math.round((count/blobs.length)*100);
                      return (
                        <div key={name} style={{display:"flex", flexDirection:"column", alignItems:"center", gap:4, flex:1}}>
                          <span style={{fontSize:11, color: MORPH_COLORS[name]||"#555b77", fontFamily:"var(--mono)"}}>{count}</span>
                          <div style={{width:"100%", height: Math.max(4, pct*0.6), background: MORPH_COLORS[name]||"#555b77", borderRadius:"3px 3px 0 0", opacity:0.85}} />
                          <span style={{fontSize:10, color:"var(--text3)", textTransform:"capitalize"}}>{name}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

            </div>
          );
        })()}

        {/* ── DASHBOARD PAGE ── */}
        {activePage === "dashboard" && (
          <div className="dash-page">
            {result
              ? <Dashboard blobs={result.blobs} imageName={imageFile?.name} />
              : <div className="viewer-empty"><div className="viewer-empty-icon">📊</div><p>Run analysis first to see the dashboard</p></div>
            }
          </div>
        )}

        {/* ── CORRELATION PAGE ── */}
        {activePage === "correlation" && (
          <div className="dash-page" style={{ display: "flex", flexDirection: "column", gap: 28, padding: 24 }}>

            {/* Autocorrelation — from the main vessel-mask pipeline */}
            <section>
              <div className="dash-chart-title" style={{ fontSize: 14, marginBottom: 12 }}>
                2D Autocorrelation — Vessel Mask
              </div>
              {!result ? (
                <div className="viewer-empty" style={{ height: 200 }}>
                  <div className="viewer-empty-icon">🧬</div>
                  <p>Run the main analysis (Viewer tab) to see the vessel mask's autocorrelation</p>
                </div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "minmax(260px, 380px) 1fr", gap: 20 }}>
                  <div className="canvas-frame" style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: 12 }}>
                    <img src={result.images.autocorrelation_2d} alt="2D autocorrelation" style={{ width: "100%", display: "block", borderRadius: 8 }} />
                    {result.reports?.autocorrelation_npz && (
                      <a
                        href={`${API_BASE}/output-file/${result.reports.autocorrelation_npz}`}
                        download
                        style={{ display: "inline-block", marginTop: 10, fontSize: 11.5, color: "var(--accent, #4f8ef7)", textDecoration: "none" }}
                      >
                        ⬇ Download raw data (.npz)
                      </a>
                    )}
                  </div>
                  <div className="dash-chart-card">
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <div className="dash-chart-title">
                        Radial Profile ({result.acf_radial?.unit_label})
                      </div>
                      {result.acf_radial?.radius?.length > 0 && (() => {
                        const fullMax = result.acf_radial.radius[result.acf_radial.radius.length - 1];
                        return (
                          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text3)" }}>
                            <span>x-range:</span>
                            <input
                              type="number"
                              value={Number.isFinite(acfXMin) ? acfXMin : 0}
                              min={0}
                              max={fullMax}
                              step="any"
                              onChange={e => setAcfXMin(e.target.value === "" ? 0 : Number(e.target.value))}
                              style={{ width: 64, background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text)", padding: "3px 6px", fontFamily: "var(--mono)" }}
                            />
                            <span>to</span>
                            <input
                              type="number"
                              value={Number.isFinite(acfXMax) ? acfXMax : Math.round(fullMax)}
                              min={0}
                              max={fullMax}
                              step="any"
                              onChange={e => setAcfXMax(e.target.value === "" ? fullMax : Number(e.target.value))}
                              style={{ width: 64, background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text)", padding: "3px 6px", fontFamily: "var(--mono)" }}
                            />
                            <span>{result.acf_radial.unit_label}</span>
                            <button
                              onClick={() => { setAcfXMin(0); setAcfXMax(fullMax); }}
                              style={{ background: "none", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text3)", padding: "3px 8px", fontSize: 10.5, cursor: "pointer" }}
                            >
                              reset
                            </button>
                          </div>
                        );
                      })()}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text3)", marginTop: -6, marginBottom: 10, lineHeight: 1.6 }}>
                      {result.acf_radial?.correlation_length != null && (
                        <span style={{ color: "#f5a623" }}>ℓc (1/e) ≈ {result.acf_radial.correlation_length.toFixed(2)} {result.acf_radial.unit_label}</span>
                      )}
                      {result.acf_radial?.half_max_distance != null && (
                        <span style={{ color: "#a78bfa", marginLeft: 14 }}>r½ (half-max) ≈ {result.acf_radial.half_max_distance.toFixed(2)} {result.acf_radial.unit_label}</span>
                      )}
                      <br />Error bars = ±1 standard deviation of ACF across all angles at each distance bin
                    </div>
                    <ResponsiveContainer width="100%" height={260}>
                      <ComposedChart margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
                        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                        <XAxis dataKey="r" type="number" domain={[acfXMin ?? 0, acfXMax ?? "dataMax"]} stroke="var(--text3)" fontSize={11}
                          label={{ value: `Distance (${result.acf_radial.unit_label})`, position: "insideBottom", offset: -4, fill: "var(--text3)", fontSize: 11 }} />
                        <YAxis stroke="var(--text3)" fontSize={11}
                          label={{ value: "ACF", angle: -90, position: "insideLeft", fill: "var(--text3)", fontSize: 11 }} />
                        <RTooltip
                          contentStyle={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                          formatter={(v, name) => name === "acf" ? [Number(v).toFixed(4), "ACF"] : null}
                          labelFormatter={(l) => `r = ${l}`} />
                        {result.acf_radial?.correlation_threshold != null && (
                          <ReferenceLine y={result.acf_radial.correlation_threshold} stroke="var(--text3)" strokeDasharray="4 4"
                            label={{ value: "1/e", position: "insideTopRight", fill: "var(--text3)", fontSize: 10 }} />
                        )}
                        {result.acf_radial?.correlation_length != null && (
                          <ReferenceLine x={Number(result.acf_radial.correlation_length.toFixed(2))} stroke="#f5a623" strokeDasharray="4 4"
                            label={{ value: "ℓc", position: "top", fill: "#f5a623", fontSize: 11 }} />
                        )}
                        {result.acf_radial?.half_max_distance != null && (
                          <ReferenceLine x={Number(result.acf_radial.half_max_distance.toFixed(2))} stroke="#a78bfa" strokeDasharray="4 4"
                            label={{ value: "r½", position: "top", fill: "#a78bfa", fontSize: 11 }} />
                        )}
                        <Line
                          type="monotone" dataKey="acf" stroke="#e63946" strokeWidth={2} dot={false} isAnimationActive={false}
                          data={result.acf_radial.radius
                            .map((r, i) => ({ r: Number(r.toFixed(2)), acf: result.acf_radial.profile[i] }))
                            .filter(d => (acfXMin == null || d.r >= acfXMin) && (acfXMax == null || d.r <= acfXMax))}
                        />
                        <Scatter
                          dataKey="acf" fill="#e63946" isAnimationActive={false}
                          data={sampleIndices(result.acf_radial.radius.length)
                            .map(i => ({
                              r: Number(result.acf_radial.radius[i].toFixed(2)),
                              acf: result.acf_radial.profile[i],
                              std: result.acf_radial.std ? result.acf_radial.std[i] : 0,
                            }))
                            .filter(d => (acfXMin == null || d.r >= acfXMin) && (acfXMax == null || d.r <= acfXMax))}
                        >
                          <ErrorBar dataKey="std" width={4} strokeWidth={1.4} stroke="#e63946" />
                        </Scatter>
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </section>

            <div style={{ borderTop: "1px solid var(--border)" }} />

            {/* Cross-correlation — fully independent of the main analysis uploads */}
            <section>
              <div className="dash-chart-title" style={{ fontSize: 14, marginBottom: 4 }}>
                2D Cross-Correlation — Two Independent Inputs
              </div>
              <p style={{ fontSize: 12, color: "var(--text3)", margin: "0 0 14px" }}>
                Upload two separate files below — each can be a regular image or a GeoJSON annotation.
                This is computed only from these two uploads, not from the image/GeoJSON used for the main analysis above.
              </p>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 12, alignItems: "end", marginBottom: 16 }}>
                <FileUpload label="Upload Input A" accept="" hint="image or .geojson — any file type shown" file={xcorrFileA} onChange={setXcorrFileA} />
                <FileUpload label="Upload Input B" accept="" hint="image or .geojson — any file type shown" file={xcorrFileB} onChange={setXcorrFileB} />
                <button className={`btn-process${xcorrLoading ? " loading" : ""}`} onClick={runCrossCorrelation} disabled={xcorrLoading || !xcorrFileA || !xcorrFileB}>
                  {xcorrLoading ? "⏳  Computing…" : "▶  Run Cross-Correlation"}
                </button>
              </div>

              {xcorrError && (
                <div style={{ color: "var(--danger, #e8607a)", fontSize: 12.5, marginBottom: 12 }}>Error: {xcorrError}</div>
              )}

              {xcorrResult && (
                <div style={{ display: "grid", gridTemplateColumns: "minmax(260px, 380px) 1fr", gap: 20 }}>
                  <div className="canvas-frame" style={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: 12 }}>
                    <img src={xcorrResult.png_data_url} alt="2D cross-correlation" style={{ width: "100%", display: "block", borderRadius: 8 }} />
                    <div style={{ fontSize: 11.5, color: "var(--text3)", marginTop: 10, lineHeight: 1.7, fontFamily: "var(--mono)" }}>
                      <div>Peak offset: Δx={xcorrResult.peak_dx} px, Δy={xcorrResult.peak_dy} px</div>
                      <div>Peak value: {xcorrResult.peak_value.toFixed(4)}</div>
                      {xcorrResult.half_max_distance != null && (
                        <div>Half-max distance (r½): {xcorrResult.half_max_distance.toFixed(2)} px</div>
                      )}
                      <div>Working size: {xcorrResult.working_shape.width} × {xcorrResult.working_shape.height} px</div>
                    </div>
                    {xcorrResult.npz_download_path && (
                      <a
                        href={`${API_BASE}/output-file/${xcorrResult.npz_download_path}`}
                        download
                        style={{ display: "inline-block", marginTop: 10, fontSize: 11.5, color: "var(--accent, #4f8ef7)", textDecoration: "none" }}
                      >
                        ⬇ Download raw data (.npz)
                      </a>
                    )}
                  </div>
                  <div className="dash-chart-card">
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <div className="dash-chart-title">Radial Profile (px)</div>
                      {xcorrResult.radius?.length > 0 && (() => {
                        const fullMax = xcorrResult.radius[xcorrResult.radius.length - 1];
                        return (
                          <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text3)" }}>
                            <span>x-range:</span>
                            <input
                              type="number"
                              value={Number.isFinite(xcorrXMin) ? xcorrXMin : 0}
                              min={0}
                              max={fullMax}
                              step="any"
                              onChange={e => setXcorrXMin(e.target.value === "" ? 0 : Number(e.target.value))}
                              style={{ width: 64, background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text)", padding: "3px 6px", fontFamily: "var(--mono)" }}
                            />
                            <span>to</span>
                            <input
                              type="number"
                              value={Number.isFinite(xcorrXMax) ? xcorrXMax : Math.round(fullMax)}
                              min={0}
                              max={fullMax}
                              step="any"
                              onChange={e => setXcorrXMax(e.target.value === "" ? fullMax : Number(e.target.value))}
                              style={{ width: 64, background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text)", padding: "3px 6px", fontFamily: "var(--mono)" }}
                            />
                            <span>px</span>
                            <button
                              onClick={() => { setXcorrXMin(0); setXcorrXMax(fullMax); }}
                              style={{ background: "none", border: "1px solid var(--border)", borderRadius: 5, color: "var(--text3)", padding: "3px 8px", fontSize: 10.5, cursor: "pointer" }}
                            >
                              reset
                            </button>
                          </div>
                        );
                      })()}
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text3)", marginTop: -6, marginBottom: 10, lineHeight: 1.6 }}>
                      {xcorrResult.half_max_distance != null && (
                        <span style={{ color: "#a78bfa" }}>r½ (half-max) ≈ {xcorrResult.half_max_distance.toFixed(2)} px</span>
                      )}
                      <br />Error bars = ±1 standard deviation across all angles at each distance bin
                    </div>
                    <ResponsiveContainer width="100%" height={260}>
                      <ComposedChart margin={{ top: 8, right: 24, bottom: 8, left: 0 }}>
                        <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                        <XAxis dataKey="r" type="number" domain={[xcorrXMin ?? 0, xcorrXMax ?? "dataMax"]} stroke="var(--text3)" fontSize={11}
                          label={{ value: "Distance (px)", position: "insideBottom", offset: -4, fill: "var(--text3)", fontSize: 11 }} />
                        <YAxis stroke="var(--text3)" fontSize={11}
                          label={{ value: "Cross-corr.", angle: -90, position: "insideLeft", fill: "var(--text3)", fontSize: 11 }} />
                        <RTooltip
                          contentStyle={{ background: "var(--bg2)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                          formatter={(v, name) => name === "xc" ? [Number(v).toFixed(4), "XC"] : null}
                          labelFormatter={(l) => `r = ${l}`} />
                        <ReferenceLine y={0} stroke="var(--text3)" strokeDasharray="2 2" />
                        {xcorrResult.half_max_distance != null && (
                          <ReferenceLine x={Number(xcorrResult.half_max_distance.toFixed(2))} stroke="#a78bfa" strokeDasharray="4 4"
                            label={{ value: "r½", position: "top", fill: "#a78bfa", fontSize: 11 }} />
                        )}
                        <Line
                          type="monotone" dataKey="xc" stroke="#4f8ef7" strokeWidth={2} dot={false} isAnimationActive={false}
                          data={xcorrResult.radius
                            .map((r, i) => ({ r: Number(r.toFixed(2)), xc: xcorrResult.profile[i] }))
                            .filter(d => (xcorrXMin == null || d.r >= xcorrXMin) && (xcorrXMax == null || d.r <= xcorrXMax))}
                        />
                        <Scatter
                          dataKey="xc" fill="#4f8ef7" isAnimationActive={false}
                          data={sampleIndices(xcorrResult.radius.length)
                            .map(i => ({
                              r: Number(xcorrResult.radius[i].toFixed(2)),
                              xc: xcorrResult.profile[i],
                              std: xcorrResult.std ? xcorrResult.std[i] : 0,
                            }))
                            .filter(d => (xcorrXMin == null || d.r >= xcorrXMin) && (xcorrXMax == null || d.r <= xcorrXMax))}
                        >
                          <ErrorBar dataKey="std" width={4} strokeWidth={1.4} stroke="#4f8ef7" />
                        </Scatter>
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </section>
          </div>
        )}
      </main>

      {/* Hover tooltip */}
      {tooltip && tooltip.blob && activePage === "viewer" && (
        <div className="tooltip" style={{ left: tooltip.left, top: tooltip.top }}>
          <div className="tooltip-id">Blob #{tooltip.blob.blob_id}</div>
          {[
            ["morphology",   tooltip.blob.morphology_2d],
            ["circularity",  tooltip.blob.circularity_2d  != null ? Number(tooltip.blob.circularity_2d).toFixed(3) : "—"],
            ["length",       tooltip.blob.length_clean    != null ? Number(tooltip.blob.length_clean).toFixed(1) + " px" : "—"],
            ["width",        tooltip.blob.width_clean     != null ? Number(tooltip.blob.width_clean).toFixed(1) + " px" : "—"],
            ["area",         tooltip.blob.area_px         != null ? Math.round(tooltip.blob.area_px).toLocaleString() + " px²" : "—"],
            ["segments",     tooltip.blob.n_segments],
          ].map(([k,v]) => (
            <div className="tooltip-row" key={k}>
              <span className="tooltip-key">{k}</span>
              <span className="tooltip-val">{v}</span>
            </div>
          ))}
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          <span className="toast-icon">{toast.type==="success" ? "✓" : "✕"}</span>
          <span className="toast-msg">{toast.msg}</span>
          <button className="toast-close" onClick={() => setToast(null)}>×</button>
        </div>
      )}
    </div>
  );
}
