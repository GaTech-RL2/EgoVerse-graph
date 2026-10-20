// Exercise actual rollout JS without connecting a camera, model, or robot.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const staticRoot = path.join(root, 'egomimic/robot/rollout_dashboard_static');
const app = fs.readFileSync(path.join(staticRoot, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(staticRoot, 'index.html'), 'utf8');
const catalog = JSON.parse(html.match(/<script id="reset-guide-catalog" type="application\/json">(.*?)<\/script>/s)[1]);
assert.equal(catalog.version,1,'Generate guides in the activated project environment before running UI tests');
assert.match(app,/function setupResetGuides\(/,'Run build_reset_guides.py to generate the shared rollout renderer');

class Element {
  constructor() {
    this.children = []; this.attrs = {}; this.value = ''; this.disabled = false;
    this.dataset = {}; this.classList = {add() {}, remove() {}, toggle() {}};
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  removeAttribute(name) { delete this.attrs[name]; }
  addEventListener() {}
  querySelectorAll() { return []; }
}
class Select extends Element {}
class Input extends Element {}
class Button extends Element {}
class TextArea extends Element {}
class Video extends Element {}
class Canvas extends Element {
  constructor() { super(); this.width = 300; this.height = 150; this.draws = []; }
  getContext() { return {drawImage: (...args) => this.draws.push(args)}; }
}

function fixture(data = catalog, controls = true) {
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!controls && id.startsWith('reset-guide')) return null;
      if (!elements.has(id)) elements.set(id, id === 'reset-guide' ? new Select() : new Element());
      return elements.get(id);
    },
    createElement(tag) { return tag === 'canvas' ? new Canvas() : new Element(); },
    createElementNS() { return new Element(); },
    createTextNode(text) { return {textContent:text}; },
    querySelectorAll() { return []; },
  };
  if (controls) document.getElementById('reset-guide-catalog').textContent = JSON.stringify(data);
  const sent = [];
  class WebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 1; }
    send(value) { sent.push(JSON.parse(value)); }
  }
  let closedBitmaps = 0;
  const context = vm.createContext({document, WebSocket, Blob,
    location:{protocol:'http:',host:'fixture'}, HTMLInputElement:Input,
    HTMLSelectElement:Select, HTMLTextAreaElement:TextArea, HTMLButtonElement:Button,
    HTMLVideoElement:Video, setTimeout() {}, clearTimeout() {}, console,
    confirm() { throw Error('A guide must never request robot control'); },
    createImageBitmap:async () => ({width:640,height:480,close() { closedBitmaps++; }})});
  vm.runInContext(app, context);
  const run = code => vm.runInContext(code, context);
  const config = `configure({cameras:['front_img_1','left_wrist_img','right_wrist_img'],wait_for_start:true,overlay_camera:'front_img_1'})`;
  const frame = (images=['front_img_1','left_wrist_img','right_wrist_img'], overlay=false) =>
    run(`frame({status:'Ready',age_ms:0,images:${JSON.stringify(images)},overlay_enabled:${overlay}})`);
  const decode = name => run(`drawCameraFrame('${name}',new Uint8Array([1,2,3]).buffer)`);
  const frames = async () => { frame(); for (const name of ['front_img_1','left_wrist_img','right_wrist_img']) await decode(name); };
  const select = id => { document.getElementById('reset-guide').value = id; document.getElementById('reset-guide').onchange(); };
  const hidden = () => run(`Object.hasOwn(cameras.get('front_img_1').overlay.attrs,'hidden')`);
  return {document,run,config,frame,decode,frames,select,hidden,sent,context,closedBitmaps:() => closedBitmaps};
}

test('same teleop templates, Off default, front-only SVG, zero robot or recording writes', async () => {
  const teleop = fs.readFileSync(path.join(root,'egomimic/robot/teleop_dashboard_static/index.html'),'utf8');
  const source = JSON.parse(teleop.match(/<script id="reset-guide-catalog" type="application\/json">(.*?)<\/script>/s)[1]);
  assert.deepEqual(catalog,source);
  const f=fixture(); f.run(f.config); await f.frames();
  assert.equal(f.hidden(),true);
  const before=f.run(`cameras.get('front_img_1').view.draws.length`);
  f.select(catalog.templates[0].id);
  assert.equal(f.hidden(),false);
  assert.equal(f.run(`cameras.get('front_img_1').overlay.children.length`),6);
  assert.equal(f.run(`cameras.get('front_img_1').overlay.attrs.viewBox`),'0 0 640 480');
  assert.equal(f.run(`cameras.get('front_img_1').overlay.attrs.preserveAspectRatio`),'xMidYMid meet');
  assert.equal(f.run(`cameras.get('front_img_1').view.draws.length`),before);
  assert.equal(f.closedBitmaps(),3);
  for (const camera of ['left_wrist_img','right_wrist_img']) assert.equal(f.run(`cameras.get('${camera}').overlay.children.length`),0);
  f.select(''); assert.equal(f.hidden(),true);
  assert.deepEqual(f.sent,[]);
});

test('independent of Cartesian action overlay, selection survives reconnect and model config', async () => {
  const f=fixture(); f.run(f.config); await f.frames(); f.select(catalog.templates[0].id);
  f.frame(undefined,true);
  assert.equal(f.document.getElementById('overlay').checked,true);
  assert.equal(f.hidden(),false);
  f.select(''); assert.equal(f.document.getElementById('overlay').checked,true);
  f.select(catalog.templates[0].id); f.run('socket.onclose({code:1006})');
  assert.equal(f.hidden(),true);
  f.run(f.config); assert.equal(f.hidden(),true);
  await f.frames(); assert.equal(f.hidden(),false);
  f.run('socket.onclose({code:4001})'); assert.equal(f.hidden(),true);
  assert.deepEqual(f.sent,[]);
});

test('missing or failed frames and incompatible camera aspect hide guide until fresh decode', async () => {
  const f=fixture(); f.run(f.config); await f.frames(); f.select(catalog.templates[0].id);
  f.frame([]); assert.equal(f.hidden(),true);
  f.frame(); assert.equal(f.hidden(),true);
  await f.decode('front_img_1'); assert.equal(f.hidden(),false);
  f.context.createImageBitmap=async () => ({width:1280,height:480,close() {}});
  await f.decode('front_img_1'); assert.equal(f.hidden(),true);
  assert.match(f.document.getElementById('reset-guide-status').textContent,/aspect ratio/);
  f.context.createImageBitmap=async () => ({width:1280,height:960,close() {}});
  await f.decode('front_img_1'); assert.equal(f.hidden(),false);
  f.context.createImageBitmap=async () => { throw Error('bad JPEG'); };
  await f.decode('front_img_1'); assert.equal(f.hidden(),true);
  assert.deepEqual(f.sent,[]);
});

test('unknown or invalid templates fail closed without affecting camera or rollout controls', async () => {
  const f=fixture(); f.run(f.config); await f.frames(); f.select('missing'); assert.equal(f.hidden(),true);
  f.select(catalog.templates[0].id); f.run(`configure({cameras:['left_wrist_img']})`);
  assert.match(f.document.getElementById('reset-guide-status').textContent,/Waiting for front_img_1/);
  const broken=fixture({version:1,templates:[{id:'broken'}]});
  assert.equal(broken.document.getElementById('reset-guide').disabled,true);
  broken.run(broken.config); await broken.frames();
  assert.match(broken.document.getElementById('reset-guide-status').textContent,/unavailable/);
  assert.equal(broken.run(`cameras.get('front_img_1').view.draws.length`),1);
  assert.deepEqual(broken.sent,[]);
});

test('template dropdown keyboard cannot start, pause, home, execute, resample or record', async () => {
  const f=fixture(); f.run(f.config); await f.frames();
  for (const key of ['c','r','v','e','s','q',' ']) {
    f.document.onkeydown({key,code:key===' '?'Space':'',target:f.document.getElementById('reset-guide')});
  }
  assert.deepEqual(f.sent,[]);
});

test('new assets tolerate an older page during static-only deployment', async () => {
  const f=fixture(catalog,false); f.run(f.config); await f.frames();
  assert.equal(f.run(`cameras.get('front_img_1').view.draws.length`),1);
  assert.deepEqual(f.sent,[]);
});
