"""Local-only aggregate training monitor. Checkpoints and source rows are never served."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_bytes()) if path.is_file() else None


def progress(root):
    runs=[]
    for recipe_path in sorted(Path(root).glob('*/recipe.json')):
        folder=recipe_path.parent;recipe=read_json(recipe_path)
        metrics=[]
        if (folder/'metrics.jsonl').is_file():
            for line in (folder/'metrics.jsonl').read_text().splitlines():
                try:metrics.append(json.loads(line))
                except json.JSONDecodeError:continue # A live writer may not yet have completed its final line.
        runs.append(dict(name=folder.name,evidence=recipe['evidence'],phase=recipe['phase'],
                         model=recipe['model'],config=recipe['train'],metrics=metrics,
                         terminal=read_json(folder/'terminal.json'),
                         checkpoint=read_json(folder/'checkpoints/latest.json')))
    return dict(format='dope-shared-training-dashboard-v1',runs=runs)


HTML='''<!doctype html><meta charset="utf-8"><title>DOPE shared model training</title>
<style>body{font:16px system-ui;background:#0d1520;color:#e3eaf3;margin:32px}h1{font-size:28px}p{color:#aebfce}.run{border:1px solid #344557;border-radius:12px;padding:20px;margin:20px 0}svg{width:100%;height:220px;background:#111f2c}small{color:#aebfce}b{color:#77dac3}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}.legend{color:#72d9ba}.val{color:#f8bc74}code{overflow-wrap:anywhere}</style>
<h1>DOPE · Shared table model</h1><p>Actual optimizer updates · TRAIN-only development episodes · No qualification or external accuracy claim</p>
<section class="run" id="precalc" hidden></section>
<div id="runs">Waiting for a run with recorded metrics…</div>
<script>
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function plot(metrics,level){let points=metrics.filter(m=>m.levels[level]&&Number.isFinite(m.levels[level].total));if(!points.length)return '<p>No observations</p>';
let xmax=Math.max(1,...points.map(p=>p.step)),ymax=Math.max(.001,...points.map(p=>p.levels[level].total)),ymin=Math.min(0,...points.map(p=>p.levels[level].total));
let lines=['train','validation'].map((kind,i)=>{let data=points.filter(p=>p.kind===kind);let path=data.map(p=>`${30+p.step/xmax*540},${190-(p.levels[level].total-ymin)/(ymax-ymin)*170}`).join(' ');return `<polyline fill="none" stroke="${i?'#f8bc74':'#72d9ba'}" stroke-width="2" points="${path}"/>`+data.map(p=>`<circle r="3" fill="${i?'#f8bc74':'#72d9ba'}" cx="${30+p.step/xmax*540}" cy="${190-(p.levels[level].total-ymin)/(ymax-ymin)*170}"/>`).join('')}).join('');
return `<svg viewBox="0 0 600 220" role="img" aria-label="${level} total objective over optimizer updates"><text x="5" y="15" fill="#aebfce">${ymax.toFixed(3)}</text>${lines}<text x="30" y="214" fill="#aebfce">0</text><text x="480" y="214" fill="#aebfce">${xmax} updates</text></svg>`;}
async function refresh(){try{let d=await(await fetch('/progress.json',{cache:'no-store'})).json();document.getElementById('runs').innerHTML=d.runs.length?d.runs.map(r=>{let last=r.metrics.filter(m=>m.kind==='train').at(-1),state=r.terminal?.state||'running / awaiting first update';let panel=r.metrics.filter(m=>m.kind==='validation'&&m.observer).at(-1)?.observer;let levels=[...new Set(r.metrics.flatMap(m=>Object.keys(m.levels)))];return `<section class="run"><h2>${esc(r.name)} · <b>${esc(state)}</b></h2><p>${esc(r.evidence)} · ${esc(r.phase)} · ${last?.optimizer_updates||0} optimizer updates · ${last?.episodes||0} episodes · ${last?.processed_cells||0} cells</p><p>Checkpoint: <code>${esc(r.checkpoint?.checkpoint||'none recorded')}</code> · GPU peak: ${last?.cuda_peak_bytes?Math.round(last.cuda_peak_bytes/1048576)+' MiB':'unavailable'}</p><p>Development panel: ${panel?`${esc(panel.recipe)} · ${esc(panel.completed_episodes)}/${esc(panel.scheduled_episodes)} episodes · ${esc(panel.unique_physical_tables)} distinct physical tables · ${esc(panel.state)}`:'physical coverage unavailable in historical metrics'}</p><div class="grid">${levels.map(l=>`<div><h3>${esc(l)} total objective</h3><small class="legend">● train</small> <small class="val">● held-out development</small>${plot(r.metrics,l)}</div>`).join('')}</div><small>Validation uses fixed development episodes without optimizer updates. Denoising/proxy objectives are not external accuracy or Elo. Curves preserve each recorded observation.</small></section>`}).join(''):'No optimizer runs recorded yet.'}catch(e){document.getElementById('runs').textContent='Monitor unavailable; inspect the local run receipts.'}}
async function refreshPrecalc(){try{
const d=await(await fetch('/precalc.json',{cache:'no-store'})).json();if(!d)return;
const panel=document.getElementById('precalc');panel.hidden=false;
const pub=d.publication,view=d.observed_feature_view;
const counts=pub?.counts;
const summary=pub?.all_keys_verified?`<p><b>S3 corpus verified</b> · ${esc(d.scheduled)} / ${esc(d.scheduled)} scheduled training-split cases accounted for</p><p>${esc(counts.ready||0)} reusable reports · ${esc(counts.unsupported||0)} unsupported · ${esc(counts.failed||0)} retained failures · ${esc(pub.readback_objects)} derived objects checked</p>`:`<p>${esc(d.global_unique_receipts||0)} / ${esc(d.scheduled)} receipts · ${esc(d.global_pending??'unknown')} pending · S3 publication can lag local computation</p>`;
const features=view?`<p><b>Observed-score encoder inputs</b> · ${esc(view.state)} · ${esc(view.done||0)} / ${esc(view.scheduled)} cases processed</p><small>Measured weak, zero and negative CV scores are retained; separate masks identify missing results. No CV reruns or model updates during derivation.</small>`:'';
panel.innerHTML=`<h2>Training-only CV landmark cache</h2><p>Inner cross-validation stays inside each training partition · No official TEST labels · Model training admission remains separate</p>${summary}${features}<details><summary>Original host receipts and transfers</summary><p>Recorded ${esc(new Date(d.unix*1000).toISOString())}. Counts below precede additive report validation recovery; reassigned unstarted cases are counted globally once.</p>${d.hosts.map(h=>`<p><b>${esc(h.host)}</b> · ${esc(h.state)} · ${esc(h.workers||0)} active workers · ${esc(h.terminal||0)}/${esc(h.scheduled||0)} original/assigned receipts · ${esc(JSON.stringify(h.counts))}</p>`).join('')}</details><small>All failures, unsupported inputs and source/split/producer hashes remain explicit. Cache completion does not establish model accuracy.</small>`
}catch(e){/* The cache collector may not have produced an aggregate yet. */}}

refresh();setInterval(refresh,5000);refreshPrecalc();setInterval(refreshPrecalc,30000);
</script>'''


def serve(root,port):
    root=Path(root).resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/':raw=HTML.encode();kind='text/html; charset=utf-8'
            elif self.path=='/precalc.json':raw=json.dumps(read_json(root/'precalc.json'),allow_nan=False).encode();kind='application/json'
            elif self.path=='/progress.json':raw=json.dumps(progress(root),allow_nan=False).encode();kind='application/json'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
        def log_message(self,*args):pass
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18767);args=parser.parse_args();serve(args.root,args.port)
