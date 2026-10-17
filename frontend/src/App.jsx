import { useState, useRef, useCallback, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_URL !== undefined
  ? import.meta.env.VITE_API_URL
  : (typeof window !== "undefined" && window.location.hostname === "localhost" ? "http://localhost:8000" : "");

const COLORS = {
  red:     "#E24B4A",
  green:   "#639922",
  amber:   "#BA7517",
  blue:    "#185FA5",
  teal:    "#1D9E75",
  muted:   "#888780",
  border:  "rgba(0,0,0,0.12)",
};

function Badge({ children, color = "gray" }) {
  const map = {
    red:   { bg:"#FCEBEB", text:"#A32D2D" },
    green: { bg:"#EAF3DE", text:"#3B6D11" },
    amber: { bg:"#FAEEDA", text:"#854F0B" },
    blue:  { bg:"#E6F1FB", text:"#185FA5" },
    gray:  { bg:"#F1EFE8", text:"#5F5E5A" },
  };
  const c = map[color] || map.gray;
  return (
    <span style={{
      display:"inline-block", padding:"2px 10px",
      borderRadius:99, fontSize:12, fontWeight:500,
      background:c.bg, color:c.text,
    }}>{children}</span>
  );
}

function UploadZone({ onFile, busy }) {
  const inputRef = useRef();
  const [drag, setDrag] = useState(false);

  const handleDrop = e => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  };

  return (
    <div
      onClick={() => !busy && inputRef.current.click()}
      onDragOver={e=>{e.preventDefault();setDrag(true)}}
      onDragLeave={()=>setDrag(false)}
      onDrop={handleDrop}
      style={{
        border:`2px dashed ${drag ? COLORS.blue : COLORS.border}`,
        borderRadius:12, padding:"2rem 1rem",
        textAlign:"center", cursor:busy?"not-allowed":"pointer",
        background: drag ? "#E6F1FB" : "var(--color-background-secondary)",
        transition:"all .2s", opacity: busy ? 0.6 : 1,
      }}
    >
      <div style={{fontSize:36, marginBottom:8}}>📷</div>
      <p style={{margin:0, color:"var(--color-text-secondary)", fontSize:14}}>
        {busy ? "Đang xử lý..." : "Kéo thả hoặc nhấn để chọn ảnh"}
      </p>
      <input ref={inputRef} type="file" accept="image/*" style={{display:"none"}}
        onChange={e => e.target.files[0] && onFile(e.target.files[0])} />
    </div>
  );
}

function PlateCard({ det, index }) {
  return (
    <div style={{
      background:"var(--color-background-primary)",
      border:`0.5px solid ${COLORS.border}`,
      borderRadius:12, padding:"1rem 1.25rem",
      display:"flex", flexDirection:"column", gap:8,
    }}>
      <div style={{display:"flex", justifyContent:"space-between", alignItems:"center"}}>
        <span style={{fontSize:13, color:"var(--color-text-secondary)"}}>Biển #{index+1}</span>
        <Badge color={det.hop_le ? "green" : "amber"}>
          {det.hop_le ? "✓ Hợp lệ TT79" : "Chưa khớp format"}
        </Badge>
      </div>
      <div style={{fontSize:26, fontWeight:700, letterSpacing:2, color:COLORS.blue, fontFamily:"monospace"}}>
        {det.plate_text || "—"}
      </div>
      <div style={{fontSize:12, color:"var(--color-text-secondary)"}}>
        OCR thô: <code style={{background:"var(--color-background-secondary)",padding:"1px 6px",borderRadius:4}}>{det.plate_raw}</code>
      </div>
      <div style={{display:"flex", gap:12, marginTop:4}}>
        <ConfBar label="YOLO" value={det.yolo_conf} color={COLORS.teal}/>
        <ConfBar label="OCR"  value={det.ocr_conf}  color={COLORS.blue}/>
      </div>
      {det.two_line && <Badge color="gray">Biển 2 dòng</Badge>}
    </div>
  );
}

function ConfBar({ label, value, color }) {
  const pct = Math.round(value * 100);
  return (
    <div style={{flex:1}}>
      <div style={{fontSize:11, color:"var(--color-text-secondary)", marginBottom:3}}>
        {label} {pct}%
      </div>
      <div style={{background:"var(--color-background-secondary)", borderRadius:99, height:5, overflow:"hidden"}}>
        <div style={{width:`${pct}%`, height:"100%", background:color, borderRadius:99, transition:"width .5s"}}/>
      </div>
    </div>
  );
}

function ViolationCard({ v, index }) {
  return (
    <div style={{
      background:"var(--color-background-primary)",
      border:`2px solid ${COLORS.red}22`,
      borderLeft:`4px solid ${COLORS.red}`,
      borderRadius:12, padding:"1rem 1.25rem",
      display:"flex", flexDirection:"column", gap:6,
    }}>
      <div style={{display:"flex", justifyContent:"space-between"}}>
        <span style={{fontWeight:500, color:COLORS.red}}>⚠ Vi phạm #{index+1}</span>
        <span style={{fontSize:12, color:"var(--color-text-secondary)"}}>{v.time}</span>
      </div>
      <div style={{fontSize:22, fontWeight:700, letterSpacing:2, color:COLORS.blue, fontFamily:"monospace"}}>
        {v.plate_text}
      </div>
      <div style={{display:"flex", gap:8, flexWrap:"wrap"}}>
        <Badge color="red">{v.vehicle_type}</Badge>
        <Badge color="gray">OCR {Math.round(v.ocr_conf*100)}%</Badge>
      </div>
      {v.evidence_url && (
        <a href={`${API_BASE}${v.evidence_url}`} target="_blank" rel="noreferrer"
          style={{fontSize:12, color:COLORS.blue, textDecoration:"none"}}>
          📎 Xem ảnh bằng chứng →
        </a>
      )}
    </div>
  );
}

function Tabs({ tabs, active, onChange }) {
  return (
    <div style={{display:"flex", gap:4, borderBottom:`0.5px solid ${COLORS.border}`, marginBottom:"1.5rem"}}>
      {tabs.map(t => (
        <button key={t.id} onClick={()=>onChange(t.id)} style={{
          padding:"8px 16px", border:"none", background:"none",
          cursor:"pointer", fontSize:14,
          color: active===t.id ? COLORS.blue : "var(--color-text-secondary)",
          borderBottom: active===t.id ? `2px solid ${COLORS.blue}` : "2px solid transparent",
          fontWeight: active===t.id ? 500 : 400,
          transition:"all .15s",
        }}>{t.label}</button>
      ))}
    </div>
  );
}

function ROICanvas({ imgSrc, onConfigSaved }) {
  const canvasRef = useRef();
  const [mode, setMode]         = useState("ROI");
  const [light, setLight]       = useState("RED");
  const [roi, setRoi]           = useState(null);
  const [lineY, setLineY]       = useState(null);
  const [drawing, setDrawing]   = useState(false);
  const startRef = useRef(null);
  const imgRef   = useRef(null);
  const scaleRef = useRef(1);

  useEffect(()=>{
    const canvas = canvasRef.current; if(!canvas) return;
    const img = new Image(); img.src = imgSrc;
    img.onload = () => {
      const maxW = canvas.parentElement.clientWidth || 600;
      const scale = Math.min(maxW / img.width, 1);
      scaleRef.current = scale;
      canvas.width  = img.width  * scale;
      canvas.height = img.height * scale;
      imgRef.current = img;
      redraw(canvas, img);
    };
  }, [imgSrc]);

  const redraw = useCallback((canvas, img) => {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if(roi){
      ctx.strokeStyle="#00CCFF"; ctx.lineWidth=2;
      ctx.strokeRect(roi.x1,roi.y1,roi.x2-roi.x1,roi.y2-roi.y1);
      ctx.fillStyle="rgba(0,200,255,0.1)";
      ctx.fillRect(roi.x1,roi.y1,roi.x2-roi.x1,roi.y2-roi.y1);
    }
    if(lineY!=null){
      ctx.strokeStyle=COLORS.red; ctx.lineWidth=3;
      ctx.beginPath(); ctx.moveTo(0,lineY); ctx.lineTo(canvas.width,lineY); ctx.stroke();
      ctx.fillStyle=COLORS.red; ctx.font="12px sans-serif";
      ctx.fillText("VẠCH DỪNG",8, lineY-6);
    }
  },[roi,lineY]);

  useEffect(()=>{
    if(canvasRef.current && imgRef.current) redraw(canvasRef.current,imgRef.current);
  },[roi,lineY,redraw]);

  const getPos = e => {
    const r = canvasRef.current.getBoundingClientRect();
    return {x: e.clientX-r.left, y: e.clientY-r.top};
  };

  const handleMouseDown = e => {
    const p = getPos(e);
    if(mode==="ROI"){setDrawing(true); startRef.current=p; setRoi({x1:p.x,y1:p.y,x2:p.x,y2:p.y});}
    else setLineY(p.y);
  };
  const handleMouseMove = e => {
    if(!drawing||mode!=="ROI") return;
    const p=getPos(e);
    setRoi(r=>({...r,x2:p.x,y2:p.y}));
  };
  const handleMouseUp = () => setDrawing(false);

  const save = () => {
    const s = scaleRef.current;
    onConfigSaved({
      light_status: light,
      roi_x1: roi ? Math.round(Math.min(roi.x1,roi.x2)/s) : 0,
      roi_y1: roi ? Math.round(Math.min(roi.y1,roi.y2)/s) : 0,
      roi_x2: roi ? Math.round(Math.max(roi.x1,roi.x2)/s) : 9999,
      roi_y2: roi ? Math.round(Math.max(roi.y1,roi.y2)/s) : 9999,
      line_y: lineY ? Math.round(lineY/s) : 0,
    });
  };

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <div style={{display:"flex",gap:8,flexWrap:"wrap",alignItems:"center"}}>
        {["ROI","LINE"].map(m=>(
          <button key={m} onClick={()=>setMode(m)} style={{
            padding:"6px 14px",borderRadius:6,border:"none",cursor:"pointer",fontSize:13,
            background: mode===m ? COLORS.blue : "var(--color-background-secondary)",
            color: mode===m ? "#fff" : "var(--color-text-primary)",
          }}>{m==="ROI"?"🟦 Vẽ vùng ROI":"📏 Kẻ vạch dừng"}</button>
        ))}
        <div style={{marginLeft:"auto",display:"flex",gap:6}}>
          {["RED","GREEN"].map(l=>(
            <button key={l} onClick={()=>setLight(l)} style={{
              padding:"6px 14px",borderRadius:6,border:"none",cursor:"pointer",fontSize:13,fontWeight:500,
              background: l==="RED" ? (light==="RED"?COLORS.red:"#F7C1C1") : (light==="GREEN"?COLORS.green:"#C0DD97"),
              color: "#fff",
            }}>{l==="RED"?"🔴 Đèn đỏ":"🟢 Đèn xanh"}</button>
          ))}
        </div>
      </div>
      <canvas ref={canvasRef}
        style={{borderRadius:8,border:`0.5px solid ${COLORS.border}`,cursor:"crosshair",width:"100%"}}
        onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}/>
      <button onClick={save} style={{
        padding:"8px 20px",border:"none",borderRadius:8,cursor:"pointer",
        background:COLORS.teal, color:"#fff", fontSize:14, fontWeight:500,
      }}>✅ Xác nhận cấu hình & chạy phân tích</button>
    </div>
  );
}

export default function App() {
  const [tab, setTab]           = useState("plate");
  const [file, setFile]         = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [busy, setBusy]         = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);
  const [config, setConfig]     = useState(null);
  const [confThreshold, setConf]= useState(0.35);
  const [apiUrl, setApiUrl]     = useState(API_BASE);

  const handleFile = f => {
    setFile(f); setResult(null); setError(null); setConfig(null);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const runPlateDetect = async () => {
    if(!file) return;
    setBusy(true); setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("conf_threshold", confThreshold);
      fd.append("ocr_confidence", 0.30);
      const res  = await fetch(`${apiUrl}/detect-plate`, {method:"POST",body:fd});
      const data = await res.json();
      if(!res.ok) throw new Error(data.detail||"Lỗi server");
      setResult({type:"plate", ...data});
    } catch(e){setError(e.message);}
    finally{setBusy(false);}
  };

  const runViolation = async cfg => {
    if(!file) return;
    setBusy(true); setError(null); setConfig(cfg);
    try {
      const fd = new FormData();
      fd.append("file", file);
      Object.entries(cfg).forEach(([k,v])=>fd.append(k,v));
      fd.append("conf_plate",    confThreshold);
      fd.append("ocr_confidence",0.30);
      const res  = await fetch(`${apiUrl}/traffic-violation`,{method:"POST",body:fd});
      const data = await res.json();
      if(!res.ok) throw new Error(data.detail||"Lỗi server");
      setResult({type:"violation",...data});
    } catch(e){setError(e.message);}
    finally{setBusy(false);}
  };

  const TABS = [
    {id:"plate",     label:"🔤 Nhận biển số"},
    {id:"violation", label:"🚦 Phát hiện vi phạm"},
  ];

  return (
    <div style={{maxWidth:760, margin:"0 auto", padding:"1.5rem 1rem", fontFamily:"sans-serif"}}>

      {/* Header */}
      <div style={{marginBottom:"1.5rem"}}>
        <h1 style={{fontSize:22,fontWeight:500,margin:"0 0 4px",color:"var(--color-text-primary)"}}>
          Traffic AI — Giám sát giao thông
        </h1>
        <p style={{margin:0,fontSize:13,color:"var(--color-text-secondary)"}}>
          YOLO + PaddleOCR · Nhận diện biển số · Phát hiện vi phạm đèn đỏ
        </p>
      </div>

      {/* API URL config */}
      <div style={{
        background:"var(--color-background-secondary)",
        border:`0.5px solid ${COLORS.border}`,
        borderRadius:8, padding:"10px 14px", marginBottom:"1.5rem",
        display:"flex", gap:8, alignItems:"center",
      }}>
        <span style={{fontSize:12,color:"var(--color-text-secondary)",whiteSpace:"nowrap"}}>API Server:</span>
        <input value={apiUrl} onChange={e=>setApiUrl(e.target.value)}
          style={{flex:1,fontSize:13,border:"none",background:"transparent",outline:"none",color:"var(--color-text-primary)"}}
          placeholder="http://localhost:8000"/>
      </div>

      <Tabs tabs={TABS} active={tab} onChange={t=>{setTab(t);setResult(null);setError(null);}}/>

      {/* Upload zone */}
      <UploadZone onFile={handleFile} busy={busy}/>

      {previewUrl && (
        <div style={{marginTop:"1rem"}}>
          <img src={previewUrl} alt="preview"
            style={{width:"100%",borderRadius:8,border:`0.5px solid ${COLORS.border}`,maxHeight:300,objectFit:"contain"}}/>
        </div>
      )}

      {/* Controls */}
      <div style={{display:"flex",gap:12,alignItems:"center",margin:"1rem 0",flexWrap:"wrap"}}>
        <div style={{display:"flex",gap:8,alignItems:"center",flex:1}}>
          <span style={{fontSize:13,color:"var(--color-text-secondary)",whiteSpace:"nowrap"}}>
            Confidence YOLO: {Math.round(confThreshold*100)}%
          </span>
          <input type="range" min="10" max="90" step="5"
            value={Math.round(confThreshold*100)}
            onChange={e=>setConf(e.target.value/100)}
            style={{flex:1}}/>
        </div>

        {tab==="plate" && (
          <button onClick={runPlateDetect} disabled={!file||busy} style={{
            padding:"8px 20px", border:"none", borderRadius:8, cursor:(!file||busy)?"not-allowed":"pointer",
            background: (!file||busy) ? "#B4B2A9" : COLORS.blue,
            color:"#fff", fontSize:14, fontWeight:500,
          }}>
            {busy ? "Đang xử lý…" : "🔍 Phân tích biển số"}
          </button>
        )}
      </div>

      {/* ROI Canvas (violation tab) */}
      {tab==="violation" && previewUrl && !busy && !result && (
        <div style={{marginTop:"1rem"}}>
          <p style={{fontSize:13,color:"var(--color-text-secondary)",margin:"0 0 .5rem"}}>
            Vẽ vùng ROI (tùy chọn) và kẻ vạch dừng, rồi xác nhận:
          </p>
          <ROICanvas imgSrc={previewUrl} onConfigSaved={runViolation}/>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          background:"#FCEBEB", border:`0.5px solid #F09595`,
          borderRadius:8, padding:"10px 14px", marginTop:"1rem",
          fontSize:13, color:COLORS.red,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Loading */}
      {busy && (
        <div style={{textAlign:"center",padding:"2rem",color:"var(--color-text-secondary)",fontSize:14}}>
          <div style={{fontSize:28,marginBottom:8}}>⚙️</div>
          Đang chạy YOLO + PaddleOCR…
        </div>
      )}

      {/* Results — Plate */}
      {result?.type==="plate" && (
        <div style={{marginTop:"1.5rem"}}>
          <div style={{
            display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1rem"
          }}>
            <h2 style={{margin:0,fontSize:16,fontWeight:500}}>Kết quả nhận biển số</h2>
            <Badge color={result.count>0?"green":"gray"}>{result.count} biển tìm thấy</Badge>
          </div>
          {result.count===0
            ? <p style={{color:"var(--color-text-secondary)",fontSize:14}}>Không phát hiện biển số nào. Thử hạ threshold.</p>
            : <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(260px,1fr))",gap:12}}>
                {result.detections.map((d,i)=><PlateCard key={i} det={d} index={i}/>)}
              </div>
          }
        </div>
      )}

      {/* Results — Violation */}
      {result?.type==="violation" && (
        <div style={{marginTop:"1.5rem"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1rem"}}>
            <h2 style={{margin:0,fontSize:16,fontWeight:500}}>Kết quả phân tích vi phạm</h2>
            <div style={{display:"flex",gap:8}}>
              <Badge color={result.light_status==="RED"?"red":"green"}>
                {result.light_status==="RED"?"🔴 Đèn đỏ":"🟢 Đèn xanh"}
              </Badge>
              <Badge color={result.total_violations>0?"red":"gray"}>
                {result.total_violations} vi phạm
              </Badge>
            </div>
          </div>

          {/* Summary stats */}
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10,marginBottom:"1rem"}}>
            {[
              {label:"Tổng phương tiện", value: result.vehicles?.length ?? 0, color: COLORS.blue},
              {label:"Trong vùng ROI",   value: result.vehicles?.filter(v=>v.in_roi).length ?? 0, color: COLORS.amber},
              {label:"Vi phạm",          value: result.total_violations, color: COLORS.red},
            ].map(s=>(
              <div key={s.label} style={{
                background:"var(--color-background-secondary)",
                borderRadius:8, padding:"10px 12px", textAlign:"center",
              }}>
                <div style={{fontSize:24,fontWeight:500,color:s.color}}>{s.value}</div>
                <div style={{fontSize:11,color:"var(--color-text-secondary)",marginTop:2}}>{s.label}</div>
              </div>
            ))}
          </div>

          {result.total_violations===0
            ? <p style={{color:"var(--color-text-secondary)",fontSize:14}}>Không ghi nhận vi phạm nào.</p>
            : <div style={{display:"flex",flexDirection:"column",gap:10}}>
                {result.violations.map((v,i)=><ViolationCard key={i} v={v} index={i}/>)}
              </div>
          }
        </div>
      )}

      {/* Footer */}
      <div style={{
        marginTop:"3rem", paddingTop:"1rem",
        borderTop:`0.5px solid ${COLORS.border}`,
        fontSize:12, color:"var(--color-text-secondary)", textAlign:"center",
      }}>
        Traffic AI · YOLO + PaddleOCR · Backend FastAPI
      </div>
    </div>
  );
}
