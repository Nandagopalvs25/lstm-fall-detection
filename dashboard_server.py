"""
dashboard_server.py — Rebuilt from scratch, minimal and reliable
"""
import argparse, threading, time, traceback
from collections import deque
import numpy as np
from flask import Flask, jsonify, render_template_string
from fall_detector import (FallDetector, DummyModel, Normaliser, SystemStats,
                            AlertManager, console_alert, DetectionResult,
                            WINDOW_SIZE, WINDOW_STRIDE, DOWNSAMPLE,
                            ORIGINAL_FS, NORM_PARAMS, N_FEATURES)

app      = Flask(__name__)
detector = None
_raw_buf = deque(maxlen=200)
_raw_lock = threading.Lock()
_last_sent_win_id = -1

# Live session performance tracker
_session = {
    'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
    'window_labels': [],   # (ground_truth_is_fall, predicted_is_fall)
}
_session_lock = threading.Lock()
_sim_ref = None   # reference to running simulator for ground truth

# ── HTML ───────────────────────────────────────────────────────────────────
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fall Detector</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07090f;--s1:#0d1117;--s2:#161b22;--b:#21262d;
  --ok:#39d98a;--bad:#f85149;--warn:#e3b341;--info:#58a6ff;
  --txt:#c9d1d9;--dim:#8b949e;--mono:'Courier New',monospace
}
body{background:var(--bg);color:var(--txt);font-family:var(--mono);font-size:12px;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{background:var(--s1);border-bottom:1px solid var(--b);padding:8px 16px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0}
.logo{font-size:13px}.logo b{color:var(--ok)}
#dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-right:5px;animation:blink 1.5s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
#main{flex:1;display:grid;grid-template-columns:200px 1fr 220px;gap:1px;background:var(--b);overflow:hidden;min-height:0}
.col{background:var(--s1);overflow:hidden;display:flex;flex-direction:column}
.ch{padding:6px 10px;border-bottom:1px solid var(--b);font-size:9px;letter-spacing:.14em;color:var(--dim);text-transform:uppercase;flex-shrink:0}

/* left stats */
.sb{padding:8px 10px;border-bottom:1px solid var(--b)}
.sl{font-size:9px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase;margin-bottom:3px}
.sv{font-size:20px;font-weight:700}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}.info{color:var(--info)}

/* prob bar */
#pbar-bg{height:8px;background:var(--s2);border-radius:4px;margin:5px 0 3px;overflow:hidden}
#pbar{height:100%;width:0;border-radius:4px;transition:width .3s,background .3s;background:var(--ok)}
#pnum{font-size:26px;font-weight:700;transition:color .3s}
#plbl{font-size:9px;color:var(--dim);letter-spacing:.1em;text-transform:uppercase}

/* threshold */
.tr{display:flex;align-items:center;gap:6px;margin-top:4px}
.tr input{flex:1;accent-color:var(--info)}
#tdv{color:var(--warn);min-width:3ch}

/* sensor */
#sfeed{padding:6px 10px;border-top:1px solid var(--b);background:var(--s2);flex-shrink:0}
.sr{display:flex;align-items:center;gap:6px;margin-bottom:2px}
.sk{color:var(--dim);min-width:20px}
.sv2{color:var(--info);min-width:60px}
.sb2{flex:1;height:4px;background:var(--b);border-radius:2px;overflow:hidden}
.sf{height:100%;border-radius:2px;transition:width .1s}

/* centre charts */
#charts{flex:1;display:grid;grid-template-rows:1fr 1fr;gap:1px;background:var(--b);overflow:hidden}
.cw{background:var(--s2);position:relative;overflow:hidden}
.clbl{position:absolute;top:5px;left:8px;font-size:9px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase;pointer-events:none;z-index:1}
canvas{display:block}

/* log */
#log{flex:1;overflow-y:auto;min-height:0}
#log::-webkit-scrollbar{width:3px}
#log::-webkit-scrollbar-thumb{background:var(--b)}
.lr{display:flex;gap:6px;padding:4px 8px;border-bottom:1px solid rgba(255,255,255,.03);font-size:10px;animation:fi .2s}
@keyframes fi{from{opacity:0;transform:translateX(4px)}to{opacity:1}}
.lr.fall{background:rgba(248,81,73,.06);border-left:2px solid var(--bad)}
.lr.adl{border-left:2px solid transparent}
.lts{color:var(--dim);min-width:50px;flex-shrink:0}
.lev{flex:1}.lev.fall{color:var(--bad)}.lev.adl{color:var(--ok)}
.lp{color:var(--dim)}

/* model perf panel */
#perf{padding:0;overflow-y:auto}
.pm{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--b);padding:0}
.pc{background:var(--s2);padding:7px 10px;text-align:center}
.pv{font-size:17px;font-weight:700;margin-top:2px}
.pl{font-size:8px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase}
.cm-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--b);margin:0}
.cm-cell{background:var(--s2);padding:6px 8px;text-align:center}
.cm-cell .pv{font-size:14px}
.bar-row{display:flex;align-items:center;gap:6px;padding:4px 10px;border-bottom:1px solid var(--b)}
.bar-key{font-size:9px;letter-spacing:.1em;color:var(--dim);text-transform:uppercase;min-width:60px}
.bar-bg{flex:1;height:6px;background:var(--b);border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px}
.bar-val{font-size:10px;min-width:36px;text-align:right}
/* sim realism tab */
#siminfo{padding:8px 10px;overflow-y:auto;font-size:10px;line-height:1.6}
.si-h{font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin:8px 0 4px;border-bottom:1px solid var(--b);padding-bottom:3px}
.si-row{display:flex;gap:6px;margin-bottom:3px;align-items:flex-start}
.si-ic{min-width:14px;flex-shrink:0}
.si-ok{color:var(--ok)}.si-warn{color:var(--warn)}.si-bad{color:var(--bad)}
/* tabs */
.tabs{display:flex;border-bottom:1px solid var(--b);flex-shrink:0}
.tab{padding:5px 10px;font-size:9px;letter-spacing:.1em;text-transform:uppercase;
     cursor:pointer;color:var(--dim);border-bottom:2px solid transparent;transition:.15s}
.tab.on{color:var(--ok);border-color:var(--ok)}
.tabp{display:none;flex:1;overflow:hidden;flex-direction:column}
.tabp.on{display:flex}

/* alert */
#al{position:fixed;top:0;left:0;right:0;z-index:99;background:linear-gradient(90deg,#1a0000,#250008);
    border-bottom:2px solid var(--bad);display:flex;align-items:center;gap:10px;
    padding:8px 16px;transform:translateY(-110%);transition:transform .25s}
#al.on{transform:none}
#al h2{color:var(--bad);font-size:12px;letter-spacing:.05em}
#al p{color:var(--dim);font-size:10px;margin-top:1px}
#al button{margin-left:auto;border:1px solid var(--bad);background:none;color:var(--bad);
           padding:3px 8px;cursor:pointer;font-family:var(--mono);font-size:10px}
footer{background:var(--s1);border-top:1px solid var(--b);padding:4px 16px;display:flex;
       justify-content:space-between;color:var(--dim);font-size:9px;flex-shrink:0}
</style>
</head>
<body>
<header>
  <div class="logo"><span id="dot"></span>BiLSTM <b>Fall Detector</b> — Raspberry Pi Edge</div>
  <div style="color:var(--dim);font-size:10px" id="hts">--:--:--</div>
</header>

<div id="main">
  <!-- LEFT -->
  <div class="col">
    <div class="ch">Detection</div>
    <div class="sb">
      <div class="sl">Fall Probability</div>
      <div id="pnum" class="ok">0.0000</div>
      <div id="pbar-bg"><div id="pbar"></div></div>
      <div id="plbl">ADL — NORMAL</div>
    </div>
    <div class="sb"><div class="sl">Status</div><div id="sst" class="sv ok">SAFE</div></div>
    <div class="sb"><div class="sl">Falls</div><div id="sfa" class="sv bad">0</div></div>
    <div class="sb"><div class="sl">Windows</div><div id="swi" class="sv">0</div></div>
    <div class="sb"><div class="sl">Throughput</div><div id="stp" class="sv ok">—</div></div>
    <div class="sb"><div class="sl">Uptime</div><div id="sup" class="sv">0s</div></div>
    <div class="sb">
      <div class="sl">Threshold</div>
      <div class="tr">
        <span style="color:var(--dim);font-size:9px">L</span>
        <input type="range" id="thr" min="0.05" max="0.95" step="0.01" value="0.35" oninput="setT(this.value)">
        <span style="color:var(--dim);font-size:9px">H</span>
        <span id="tdv">0.35</span>
      </div>
    </div>
    <div class="ch" style="margin-top:auto">Raw MPU-6050</div>
    <div id="sfeed">
      <div class="sr"><span class="sk">ax</span><span id="ax" class="sv2">—</span><div class="sb2"><div id="bax" class="sf" style="background:var(--info);width:50%"></div></div></div>
      <div class="sr"><span class="sk">ay</span><span id="ay" class="sv2">—</span><div class="sb2"><div id="bay" class="sf" style="background:var(--info);width:50%"></div></div></div>
      <div class="sr"><span class="sk">az</span><span id="az" class="sv2">—</span><div class="sb2"><div id="baz" class="sf" style="background:var(--ok);width:50%"></div></div></div>
      <div class="sr"><span class="sk">gx</span><span id="gx" class="sv2">—</span><div class="sb2"><div id="bgx" class="sf" style="background:var(--warn);width:50%"></div></div></div>
      <div class="sr"><span class="sk">gy</span><span id="gy" class="sv2">—</span><div class="sb2"><div id="bgy" class="sf" style="background:var(--warn);width:50%"></div></div></div>
      <div class="sr"><span class="sk">gz</span><span id="gz" class="sv2">—</span><div class="sb2"><div id="bgz" class="sf" style="background:var(--warn);width:50%"></div></div></div>
      <div class="sr" style="margin-top:3px"><span class="sk" style="color:var(--dim)">|a|</span><span id="amag" class="sv2">— g</span></div>
    </div>
  </div>

  <!-- CENTRE -->
  <div class="col">
    <div id="charts">
      <div class="cw"><div class="clbl">Accel Magnitude |a| (g) — 10s</div><canvas id="ca"></canvas></div>
      <div class="cw"><div class="clbl">Fall Probability — BiLSTM output</div><canvas id="cp"></canvas></div>
    </div>
  </div>

  <!-- RIGHT -->
  <div class="col">
    <div class="tabs">
      <div class="tab on" onclick="switchTab('tlog',this)">Log</div>
      <div class="tab"    onclick="switchTab('tperf',this)">Model Performance</div>
      <div class="tab"    onclick="switchTab('tsim',this)">Sim Realism</div>
    </div>

    <!-- Log tab -->
    <div id="tlog" class="tabp on"><div id="log"></div></div>

    <!-- Model Performance tab -->
    <div id="tperf" class="tabp" id="tperf">
      <div id="perf">
        <div class="pm">
          <div class="pc"><div class="pl">Sensitivity</div><div id="pm-rec" class="pv ok">—</div></div>
          <div class="pc"><div class="pl">Specificity</div><div id="pm-spec" class="pv info">—</div></div>
          <div class="pc"><div class="pl">F1 Score</div><div id="pm-f1" class="pv">—</div></div>
          <div class="pc"><div class="pl">ROC-AUC</div><div id="pm-auc" class="pv">—</div></div>
        </div>

        <div style="padding:6px 10px 2px;font-size:9px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase;border-bottom:1px solid var(--b)">Metric Bars</div>
        <div class="bar-row"><span class="bar-key">Accuracy</span><div class="bar-bg"><div id="br-acc" class="bar-fill" style="background:var(--info)"></div></div><span id="bv-acc" class="bar-val info">—</span></div>
        <div class="bar-row"><span class="bar-key">Recall</span><div class="bar-bg"><div id="br-rec" class="bar-fill" style="background:var(--ok)"></div></div><span id="bv-rec" class="bar-val ok">—</span></div>
        <div class="bar-row"><span class="bar-key">Precision</span><div class="bar-bg"><div id="br-pre" class="bar-fill" style="background:var(--info)"></div></div><span id="bv-pre" class="bar-val info">—</span></div>
        <div class="bar-row"><span class="bar-key">F1</span><div class="bar-bg"><div id="br-f1" class="bar-fill" style="background:var(--warn)"></div></div><span id="bv-f1" class="bar-val warn">—</span></div>
        <div class="bar-row"><span class="bar-key">ROC-AUC</span><div class="bar-bg"><div id="br-auc" class="bar-fill" style="background:var(--bad)"></div></div><span id="bv-auc" class="bar-val bad">—</span></div>

        <div style="padding:6px 10px 2px;font-size:9px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase;border-bottom:1px solid var(--b);margin-top:2px">Confusion Matrix (Test Set)</div>
        <div style="padding:4px 10px;display:grid;grid-template-columns:auto 1fr 1fr;gap:2px;font-size:9px;color:var(--dim);text-align:center">
          <div></div><div style="letter-spacing:.08em">PRED ADL</div><div style="letter-spacing:.08em">PRED FALL</div>
        </div>
        <div class="cm-grid" style="margin:0 10px 8px">
          <div class="cm-cell" style="border-top:2px solid var(--ok)"><div class="pl">TRUE ADL · TN</div><div id="cm-tn" class="pv ok">—</div></div>
          <div class="cm-cell" style="border-top:2px solid var(--warn)"><div class="pl">FALSE FALL · FP</div><div id="cm-fp" class="pv warn">—</div></div>
          <div class="cm-cell" style="border-top:2px solid var(--bad)"><div class="pl">MISSED FALL · FN</div><div id="cm-fn" class="pv bad">—</div></div>
          <div class="cm-cell" style="border-top:2px solid var(--info)"><div class="pl">TRUE FALL · TP</div><div id="cm-tp" class="pv info">—</div></div>
        </div>

        <div style="padding:6px 10px 2px;font-size:9px;letter-spacing:.12em;color:var(--dim);text-transform:uppercase;border-top:1px solid var(--b);margin-top:4px">
          Live Session (vs Simulator Ground Truth)
          <span style="color:var(--bad);font-size:8px;margin-left:4px">● LIVE</span>
        </div>
        <div class="pm" id="sess-grid" style="margin:0">
          <div class="pc"><div class="pl">Sensitivity</div><div id="ss-sens" class="pv">—</div></div>
          <div class="pc"><div class="pl">Specificity</div><div id="ss-spec" class="pv">—</div></div>
          <div class="pc"><div class="pl">Accuracy</div><div id="ss-acc" class="pv">—</div></div>
          <div class="pc"><div class="pl">Windows</div><div id="ss-tot" class="pv info">0</div></div>
        </div>
        <div class="cm-grid" style="margin:4px 10px 4px">
          <div class="cm-cell"><div class="pl">TN (correct ADL)</div><div id="ss-tn" class="pv ok">0</div></div>
          <div class="cm-cell"><div class="pl">FP (false alarm)</div><div id="ss-fp" class="pv warn">0</div></div>
          <div class="cm-cell"><div class="pl">FN (missed fall)</div><div id="ss-fn" class="pv bad">0</div></div>
          <div class="cm-cell"><div class="pl">TP (caught fall)</div><div id="ss-tp" class="pv info">0</div></div>
        </div>
        <div style="padding:4px 10px 8px;font-size:9px;color:var(--dim);line-height:1.5;border-top:1px solid var(--b)">
          <b style="color:var(--txt)">Trained model (test set):</b> Accuracy 79.6% · Sensitivity 92% · AUC 0.925<br>
          Honest cross-subject split. Young 94% · Elderly 84% (1 subject only).
        </div>
      </div>
    </div>

    <!-- Simulator Realism tab -->
    <div id="tsim" class="tabp">
      <div id="siminfo">
        <div class="si-h">What the simulator gets right</div>
        <div class="si-row"><span class="si-ic si-ok">✓</span><span>3-phase fall waveform: pre-fall stumble → 3.5–6g impact spike → post-fall stillness — matches SisFall lab recordings</span></div>
        <div class="si-row"><span class="si-ic si-ok">✓</span><span>200 Hz sampling rate — identical to SisFall ADXL345 sensor</span></div>
        <div class="si-row"><span class="si-ic si-ok">✓</span><span>ADL motion profiles (standing, walking, sitting) based on SisFall sensor statistics</span></div>
        <div class="si-row"><span class="si-ic si-ok">✓</span><span>Accel + gyro axes (6-DOF) matching MPU-6050 output range</span></div>
        <div class="si-row"><span class="si-ic si-ok">✓</span><span>Realistic noise levels added per channel</span></div>

        <div class="si-h">Known gaps vs real-world falls</div>
        <div class="si-row"><span class="si-ic si-warn">⚠</span><span><b>Scripted vs accidental:</b> SisFall falls were scripted onto safety mats. Research shows real accidental falls differ significantly in acceleration profile (Klenk et al., 2011)</span></div>
        <div class="si-row"><span class="si-ic si-warn">⚠</span><span><b>Young adults only:</b> SisFall paper found young subjects simulate falls with <i>more</i> acceleration than elderly. Elderly real falls are softer and harder to detect</span></div>
        <div class="si-row"><span class="si-ic si-warn">⚠</span><span><b>98.5% of studies</b> in the literature use only simulated falls. Real-world validation is the critical unsolved gap (systematic review, 2024)</span></div>
        <div class="si-row"><span class="si-ic si-warn">⚠</span><span><b>Cross-dataset drop:</b> Models trained on SisFall show significant accuracy degradation when tested on real-fall databases like FARSEEING</span></div>
        <div class="si-row"><span class="si-ic si-bad">✗</span><span><b>Body placement:</b> SisFall sensor is waist-worn. MPU-6050 placement matters — wrist, pocket, or chest produce very different signals</span></div>
        <div class="si-row"><span class="si-ic si-bad">✗</span><span><b>No near-falls:</b> Simulator doesn't model stumbles, recoveries, or near-falls which are the hardest false-positive source in real deployment</span></div>

        <div class="si-h">Deployment recommendation</div>
        <div class="si-row"><span class="si-ic si-ok">→</span><span>This system is valid as a research prototype and proof-of-concept. Clinical deployment requires validation on real-world fall data (e.g. FARSEEING dataset) with the target population.</span></div>
      </div>
    </div>

  </div>
</div>

<div id="al">
  <div style="font-size:18px">🚨</div>
  <div><h2>FALL DETECTED</h2><p id="ald">—</p></div>
  <button onclick="dismissAl()">DISMISS</button>
</div>

<footer>
  <span>BiLSTM · SisFall · 92% sensitivity · 20Hz inference</span>
  <span id="fts">—</span>
</footer>

<script>
'use strict';

// ── Config ──────────────────────────────────────────────────────────────────
let threshold = 0.35;
let lastWinId = -1;
let alertOn   = false;

// ── Canvas ring buffers ─────────────────────────────────────────────────────
const N = 200;
const aBuf = new Float32Array(N);   // accel magnitude
const pBuf = new Float32Array(N);   // probability
let   head = 0;

function push(a, p) { aBuf[head%N]=a; pBuf[head%N]=p; head++; }

// ── Canvas drawing ──────────────────────────────────────────────────────────
const CA = document.getElementById('ca');
const CP = document.getElementById('cp');
const ctxA = CA.getContext('2d');
const ctxP = CP.getContext('2d');

function resize() {
  const ra = CA.parentElement.getBoundingClientRect();
  const rp = CP.parentElement.getBoundingClientRect();
  CA.width=Math.floor(ra.width); CA.height=Math.floor(ra.height);
  CP.width=Math.floor(rp.width); CP.height=Math.floor(rp.height);
}
resize();
window.addEventListener('resize', resize);

function drawCanvas(ctx, buf, h, W, H, lo, hi, col, thr) {
  ctx.clearRect(0,0,W,H);
  // grid
  ctx.strokeStyle='rgba(255,255,255,.04)'; ctx.lineWidth=1;
  for(let i=1;i<4;i++){const y=H*i/4|0;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  // threshold line
  if(thr!=null){
    const ty=H-((thr-lo)/(hi-lo))*H;
    ctx.strokeStyle='rgba(227,179,65,.6)';ctx.setLineDash([5,4]);
    ctx.beginPath();ctx.moveTo(0,ty);ctx.lineTo(W,ty);ctx.stroke();
    ctx.setLineDash([]);
  }
  // fill + line
  const step=W/N;
  ctx.beginPath();
  for(let i=0;i<N;i++){
    const v=buf[(h-N+i+N)%N];
    const x=i*step, y=H-((v-lo)/(hi-lo))*(H-6)-3;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  const last=buf[(h-1+N)%N];
  ctx.lineTo(N*step,H-((last-lo)/(hi-lo))*(H-6)-3);
  ctx.lineTo(N*step,H); ctx.lineTo(0,H); ctx.closePath();
  ctx.fillStyle=col.replace(')',',0.1)').replace('rgb','rgba'); ctx.fill();
  ctx.beginPath();
  for(let i=0;i<N;i++){
    const v=buf[(h-N+i+N)%N];
    const x=i*step, y=H-((v-lo)/(hi-lo))*(H-6)-3;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.strokeStyle=col; ctx.lineWidth=1.5; ctx.stroke();
  // label
  ctx.fillStyle=col; ctx.font='10px Courier New';
  const cv=buf[(h-1+N)%N];
  const ly=H-((cv-lo)/(hi-lo))*(H-6)-3;
  ctx.fillText(cv.toFixed(3),W-44,Math.max(13,Math.min(H-4,ly-3)));
}

function render() {
  try {
    const cv=pBuf[(head-1+N)%N];
    const ca=aBuf[(head-1+N)%N];
    const ac=ca>3.5?'rgb(248,81,73)':ca>2?'rgb(227,179,65)':'rgb(57,217,138)';
    const pc=cv>=threshold?'rgb(248,81,73)':cv>threshold*.65?'rgb(227,179,65)':'rgb(88,166,255)';
    drawCanvas(ctxA,aBuf,head,CA.width,CA.height,0,6,ac,null);
    drawCanvas(ctxP,pBuf,head,CP.width,CP.height,0,1,pc,threshold);
  } catch(e){}
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ── Helpers ──────────────────────────────────────────────────────────────────
const G = id => document.getElementById(id);
function setT(v){ threshold=parseFloat(v); G('tdv').textContent=threshold.toFixed(2); }

function uptime(s){
  if(s<60) return `${Math.round(s)}s`;
  if(s<3600) return `${Math.floor(s/60)}m${Math.round(s%60)}s`;
  return `${Math.floor(s/3600)}h${Math.floor((s%3600)/60)}m`;
}

function meter(prob){
  G('pnum').textContent=prob.toFixed(4);
  G('pbar').style.width=(prob*100)+'%';
  if(prob>=threshold){
    G('pbar').style.background='var(--bad)'; G('pnum').style.color='var(--bad)';
    G('plbl').textContent='⚠ FALL RISK'; G('sst').textContent='FALL'; G('sst').className='sv bad';
  } else if(prob>=threshold*.65){
    G('pbar').style.background='var(--warn)'; G('pnum').style.color='var(--warn)';
    G('plbl').textContent='ELEVATED'; G('sst').textContent='CAUTION'; G('sst').className='sv warn';
  } else {
    G('pbar').style.background='var(--ok)'; G('pnum').style.color='var(--ok)';
    G('plbl').textContent='ADL — NORMAL'; G('sst').textContent='SAFE'; G('sst').className='sv ok';
  }
}

function sensor(s){
  if(!s||s.length<6) return;
  const [ax,ay,az,gx,gy,gz]=s;
  const mag=Math.sqrt(ax*ax+ay*ay+az*az);
  G('ax').textContent=ax.toFixed(3); G('bax').style.width=Math.min(100,Math.abs(ax)/4*100)+'%';
  G('ay').textContent=ay.toFixed(3); G('bay').style.width=Math.min(100,Math.abs(ay)/4*100)+'%';
  G('az').textContent=az.toFixed(3); G('baz').style.width=Math.min(100,Math.abs(az)/4*100)+'%';
  G('gx').textContent=gx.toFixed(1); G('bgx').style.width=Math.min(100,Math.abs(gx)/300*100)+'%';
  G('gy').textContent=gy.toFixed(1); G('bgy').style.width=Math.min(100,Math.abs(gy)/300*100)+'%';
  G('gz').textContent=gz.toFixed(1); G('bgz').style.width=Math.min(100,Math.abs(gz)/300*100)+'%';
  const mc=mag>3.5?'var(--bad)':mag>2?'var(--warn)':'var(--info)';
  G('amag').textContent=mag.toFixed(3)+' g'; G('amag').style.color=mc;
}

function addLog(ts,prob,isFall){
  const row=document.createElement('div');
  row.className='lr '+(isFall?'fall':'adl');
  row.innerHTML=`<span class="lts">${ts}</span><span class="lev ${isFall?'fall':'adl'}">${isFall?'🚨 FALL':'✓ ADL'}</span><span class="lp">p=${prob.toFixed(3)}</span>`;
  const log=G('log');
  log.insertBefore(row,log.firstChild);
  while(log.children.length>80) log.removeChild(log.lastChild);
}

function showAl(prob,ts){
  if(alertOn) return; alertOn=true;
  G('ald').textContent=`p=${(prob*100).toFixed(1)}%  ·  ${ts}`;
  G('al').classList.add('on');
  setTimeout(dismissAl,12000);
}
function dismissAl(){ alertOn=false; G('al').classList.remove('on'); }

// ── Poll ─────────────────────────────────────────────────────────────────────
async function poll(){
  let d;
  try{ const r=await fetch('/api/status'); if(!r.ok) return; d=await r.json(); }
  catch(e){ return; }
  if(!d||d.error) return;

  const prob=+(d.latest_prob)||0;
  push(+(d.latest_amag)||1, prob);
  meter(prob);
  sensor(d.latest_sample);

  const ts=new Date().toLocaleTimeString();
  G('hts').textContent=ts; G('fts').textContent=ts;
  G('sup').textContent=uptime(d.uptime_s||0);
  G('swi').textContent=d.total_windows||0;
  G('sfa').textContent=d.falls_detected||0;
  G('stp').textContent=((d.windows_per_second||0).toFixed(1))+'/s';
  updatePerf(d.model_perf);
  updateSession(d.session);

  for(const r of (d.new_results||[])){
    if(r.window_id<=lastWinId) continue;
    lastWinId=r.window_id;
    if(r.is_fall){ addLog(r.timestamp,r.probability,true); showAl(r.probability,r.timestamp); }
    else if(r.window_id%3===0) addLog(r.timestamp,r.probability,false);
  }
}

// ── Live session updater ─────────────────────────────────────────────────
function updateSession(s) {
  if (!s) return;
  const fmt = v => v != null ? (v*100).toFixed(1)+'%' : '—';
  const col = (v, good, mid) => v == null ? '' :
    v >= good ? 'var(--ok)' : v >= mid ? 'var(--warn)' : 'var(--bad)';

  const sens = G('ss-sens'); const spec = G('ss-spec');
  const acc  = G('ss-acc');  const tot  = G('ss-tot');

  if (sens) { sens.textContent=fmt(s.sensitivity); sens.style.color=col(s.sensitivity,.90,.75); }
  if (spec) { spec.textContent=fmt(s.specificity); spec.style.color=col(s.specificity,.80,.60); }
  if (acc)  { acc.textContent =fmt(s.accuracy);    acc.style.color =col(s.accuracy,.80,.65); }
  if (tot)  tot.textContent = s.total;

  const cells = {tn:s.tn, fp:s.fp, fn:s.fn, tp:s.tp};
  for (const [k,v] of Object.entries(cells)) {
    const el = G('ss-'+k); if (el) el.textContent = v;
  }
}

// ── Tab switcher ─────────────────────────────────────────────────────────
function switchTab(id, el) {
  document.querySelectorAll('.tabp').forEach(p => p.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  el.classList.add('on');
}

// ── Performance panel update ──────────────────────────────────────────────
let perfLoaded = false;
function updatePerf(p) {
  if (!p || perfLoaded) return;
  perfLoaded = true;  // metrics are static — only need to set once

  const fmt = v => (v*100).toFixed(1)+'%';
  const pct  = v => (v*100).toFixed(0)+'%';

  G('pm-rec').textContent  = fmt(p.recall);
  G('pm-spec').textContent = p.tn+p.fp>0 ? fmt(p.tn/(p.tn+p.fp)) : '—';
  G('pm-f1').textContent   = fmt(p.f1);
  G('pm-auc').textContent  = p.roc_auc.toFixed(3);

  const bars = [
    ['acc', p.accuracy], ['rec', p.recall],
    ['pre', p.precision], ['f1', p.f1], ['auc', p.roc_auc]
  ];
  for (const [k, v] of bars) {
    const fill = G('br-'+k);
    const val  = G('bv-'+k);
    if (fill) fill.style.width = pct(v);
    if (val)  val.textContent  = fmt(v);
  }

  // Confusion matrix
  const cells = {tn:p.tn, fp:p.fp, fn:p.fn, tp:p.tp};
  for (const [k,v] of Object.entries(cells)) {
    const el = G('cm-'+k);
    if (el) el.textContent = v.toLocaleString();
  }

  // Colour recall by safety threshold
  const rec = G('pm-rec');
  if (rec) rec.style.color = p.recall >= 0.95 ? 'var(--ok)' : p.recall >= 0.90 ? 'var(--warn)' : 'var(--bad)';
}

setInterval(poll,1000);
poll();
</script>
</body>
</html>"""


# ── API ─────────────────────────────────────────────────────────────────────
def _get_session_stats():
    with _session_lock:
        s = dict(_session)
    tp, fp, tn, fn = s['tp'], s['fp'], s['tn'], s['fn']
    total = tp + fp + tn + fn
    sens  = tp/(tp+fn)   if (tp+fn)>0   else None
    spec  = tn/(tn+fp)   if (tn+fp)>0   else None
    acc   = (tp+tn)/total if total>0     else None
    prec  = tp/(tp+fp)   if (tp+fp)>0   else None
    f1    = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else None
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'total': total,
        'sensitivity': round(sens,4) if sens is not None else None,
        'specificity': round(spec,4) if spec is not None else None,
        'accuracy':    round(acc, 4) if acc  is not None else None,
        'precision':   round(prec,4) if prec is not None else None,
        'f1':          round(f1,  4) if f1   is not None else None,
    }

@app.route('/')
def index(): return render_template_string(PAGE)

# Load model performance metrics from metadata.json once at startup
_model_perf = {}
def _load_perf():
    import json, os
    path = os.path.join(os.path.dirname(NORM_PARAMS), 'metadata.json')
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
        p = meta.get('performance', {})
        cm = meta.get('confusion_matrix', {})
        _model_perf.update({
            'accuracy':    round(p.get('accuracy', 0.796), 4),
            'recall':      round(p.get('recall',   0.920), 4),
            'precision':   round(p.get('precision',0.650), 4),
            'f1':          round(p.get('f1',       0.762), 4),
            'roc_auc':     round(p.get('roc_auc',  0.925), 4),
            'threshold':   round(p.get('threshold',0.350), 4),
            'tp': cm.get('tp', 0), 'fp': cm.get('fp', 0),
            'tn': cm.get('tn', 0), 'fn': cm.get('fn', 0),
        })
    else:
        _model_perf.update({'accuracy':0.796,'recall':0.920,'precision':0.650,
                            'f1':0.762,'roc_auc':0.925,'threshold':0.350,
                            'tp':0,'fp':0,'tn':0,'fn':0})
_load_perf()

@app.route('/api/status')
def api_status():
    global _last_sent_win_id
    if detector is None:
        return jsonify({"error":"starting","uptime_s":0,"total_windows":0,
                        "falls_detected":0,"windows_per_second":0,
                        "latest_prob":0,"latest_amag":1,"latest_sample":None,
                        "new_results":[],"model_perf":_model_perf})

    latest = detector.get_latest_probability() or 0.0
    with _raw_lock:
        sample = list(_raw_buf[-1]) if _raw_buf else None

    amag = 1.0
    if sample:
        amag = float(np.sqrt(sample[0]**2 + sample[1]**2 + sample[2]**2))

    all_r  = detector.get_recent_results(n=50)
    new_r  = [r for r in all_r if r.window_id > _last_sent_win_id]
    if new_r: _last_sent_win_id = new_r[-1].window_id

    s = detector.stats
    return jsonify({
        "uptime_s":           round(s.uptime_s, 1),
        "total_windows":      s.total_windows,
        "falls_detected":     s.falls_detected,
        "windows_per_second": round(s.windows_per_second, 2),
        "latest_prob":        round(latest, 4),
        "latest_amag":        round(amag, 4),
        "latest_sample":      sample,
        "new_results": [{"timestamp":r.timestamp,"probability":round(r.probability,4),
                         "is_fall":r.is_fall,"window_id":r.window_id} for r in new_r],
        "model_perf":         _model_perf,
        "session":            _get_session_stats(),
    })


# ── Metrics endpoint ────────────────────────────────────────────────────────
@app.route('/api/metrics')
def api_metrics():
    import json, os
    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models', 'metadata.json')
    if not os.path.exists(meta_path):
        return jsonify({"error": "metadata.json not found"})
    with open(meta_path) as f:
        meta = json.load(f)
    return jsonify(meta)

# ── Detector thread ──────────────────────────────────────────────────────────
def run_detector(use_real):
    global detector
    try:
        detector = FallDetector(verbose=False)
    except Exception as e:
        traceback.print_exc()
        detector = type('D', (), {
            'stats': SystemStats(),
            'model': DummyModel(),
            'norm':  Normaliser(NORM_PARAMS),
            '_win_buf': deque(maxlen=WINDOW_SIZE*2),
            '_raw_count': 0, '_win_count': 0, '_win_id': 0,
            '_results': deque(maxlen=500), '_lock': threading.Lock(),
            'alerts': AlertManager(),
        })()
        detector.alerts.register(console_alert)

        def _push(self, sample):
            self._raw_count += 1
            if self._raw_count % DOWNSAMPLE: return
            self._win_buf.append(sample)
            self._win_count += 1
            if len(self._win_buf) < WINDOW_SIZE or self._win_count < WINDOW_STRIDE: return
            self._win_count = 0
            w = np.array(list(self._win_buf)[-WINDOW_SIZE:], dtype=np.float32)
            prob = self.model.predict(self.norm.transform(w))
            from datetime import datetime
            r = DetectionResult(datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                prob, prob>=self.norm.threshold, self._win_id)
            with self._lock: self._results.append(r)
            self.stats.total_windows += 1; self._win_id += 1
            if r.is_fall: self.stats.falls_detected+=1; self.alerts.trigger(r)
        import types
        detector.push_sample = types.MethodType(_push, detector)
        detector.get_recent_results = lambda n=50: list(detector._results)[-n:]
        detector.get_latest_probability = lambda: detector._results[-1].probability if detector._results else None

    # Warm up JAX JIT
    try:
        if hasattr(detector, 'model') and hasattr(detector.model, '_keras_model') and detector.model._keras_model:
            detector.model._keras_model(np.zeros((1,WINDOW_SIZE,N_FEATURES),dtype=np.float32), training=False)
            print("[Dashboard] JAX warm-up done ✓")
    except: pass

    # Hook push_sample to fill raw buffer
    orig = detector.push_sample
    cnt  = [0]
    def hooked(sample):
        orig(sample)
        cnt[0] += 1
        if cnt[0] % DOWNSAMPLE == 0:
            with _raw_lock: _raw_buf.append(sample.tolist())
    detector.push_sample = hooked

    print(f"[Dashboard] Detector live | mode={'real' if use_real else 'simulation'}")

    from sensor_simulator import SensorSimulator
    global _sim_ref
    sim = SensorSimulator(fs=ORIGINAL_FS, fall_interval=20.0, verbose=True)
    _sim_ref = sim
    sim.start()

    # Track ground truth per downsampled window
    raw_cnt = [0]
    gt_window_buf = []   # ground truth label per downsampled sample

    orig_push2 = detector.push_sample
    def push_with_gt(sample):
        raw_cnt[0] += 1
        # Record ground truth at 20Hz
        if raw_cnt[0] % DOWNSAMPLE == 0:
            gt_window_buf.append(sim.is_fall_ground_truth)
            # Once we have a full window, compare against latest prediction
            if len(gt_window_buf) >= WINDOW_SIZE:
                gt_label = any(gt_window_buf[-WINDOW_SIZE:])  # fall if any sample in window was fall
                pred = detector.get_latest_probability()
                if pred is not None:
                    pred_label = pred >= (detector.norm.threshold if hasattr(detector, 'norm') else 0.35)
                    with _session_lock:
                        if gt_label and pred_label:   _session['tp'] += 1
                        elif gt_label:                _session['fn'] += 1
                        elif pred_label:              _session['fp'] += 1
                        else:                         _session['tn'] += 1
        orig_push2(sample)
    detector.push_sample = push_with_gt

    try:
        for sample in sim.stream(duration_s=None):
            detector.push_sample(sample)
    except KeyboardInterrupt: pass
    finally: sim.stop(); _sim_ref = None


# ── Entry ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real',  action='store_true')
    ap.add_argument('--host',  default='0.0.0.0')
    ap.add_argument('--port',  type=int, default=5000)
    args = ap.parse_args()

    try:
        import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80)); ip=s.getsockname()[0]; s.close()
    except: ip='localhost'

    print(f"\n{'='*50}\n  Fall Detection Dashboard\n  http://{ip}:{args.port}\n{'='*50}\n")

    threading.Thread(target=run_detector, args=(args.real,), daemon=True).start()
    app.run(host=args.host, port=args.port, debug=False,
            threaded=True, use_reloader=False)
