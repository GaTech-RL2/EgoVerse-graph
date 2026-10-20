// Hardware-free tests of the actual dashboard JS. Run: node --test tests/test_reset_guides.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
const staticRoot = path.join(root, 'egomimic/robot/teleop_dashboard_static');
const app = fs.readFileSync(path.join(staticRoot, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(staticRoot, 'index.html'), 'utf8');
const catalog = JSON.parse(html.match(/<script id="reset-guide-catalog" type="application\/json">(.*?)<\/script>/s)[1]);

class Element {
  constructor() {
    this.children = []; this.attrs = {}; this.value = ''; this.disabled = false;
    this.naturalWidth = 640; this.naturalHeight = 480;
    this.classList = {add() {}, remove() {}};
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(name, value) { this.attrs[name] = String(value); }
  removeAttribute(name) { delete this.attrs[name]; }
}
class Select extends Element {}
class Input extends Element {}
class TextArea extends Element {}

function fixture(data = catalog) {
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, id === 'reset-guide' ? new Select() : new Element());
      return elements.get(id);
    },
    createElement(tag) { return tag === 'select' ? new Select() : new Element(); },
    createElementNS() { return new Element(); },
    querySelectorAll() { return []; },
  };
  document.getElementById('reset-guide-catalog').textContent = JSON.stringify(data);
  const sent = [];
  class WebSocket {
    static OPEN = 1;
    constructor() { this.readyState = 1; }
    send(value) { sent.push(JSON.parse(value)); }
  }
  const context = vm.createContext({document, WebSocket, location: {protocol: 'http:', host: 'fixture'},
    HTMLElement: Element, HTMLInputElement: Input, HTMLSelectElement: Select, HTMLTextAreaElement: TextArea,
    setTimeout() {}, console});
  vm.runInContext(app, context);
  const run = code => vm.runInContext(code, context);
  const config = `configure({cameras:['front_img_1','left_wrist_img','right_wrist_img'], keys:{home:'y',record:'b',activate_both:'g'}})`;
  const frames = `frame({status:'disarmed',age_ms:0,images:{front_img_1:'unchanged-jpeg',left_wrist_img:'left',right_wrist_img:'right'}}); for (const t of cameras.values()) t.image.onload()`;
  const select = id => { document.getElementById('reset-guide').value = id; document.getElementById('reset-guide').onchange(); };
  const hidden = () => run(`Object.hasOwn(cameras.get('front_img_1').overlay.attrs, 'hidden')`);
  return {document, run, config, frames, select, hidden, sent};
}

test('Off default, front-only projection, camera bytes unchanged, zero commands', () => {
  const f = fixture(); f.run(f.config); f.run(f.frames);
  assert.equal(f.hidden(), true);
  f.select(catalog.templates[0].id);
  assert.equal(f.hidden(), false);
  assert.equal(f.run(`cameras.get('front_img_1').overlay.children.length`), 6);
  assert.equal(f.run(`cameras.get('front_img_1').overlay.attrs.viewBox`), '0 0 640 480');
  assert.equal(f.run(`cameras.get('front_img_1').overlay.attrs.preserveAspectRatio`), 'xMidYMid meet');
  assert.equal(f.run(`cameras.get('front_img_1').image.src`), 'unchanged-jpeg');
  for (const camera of ['left_wrist_img', 'right_wrist_img']) {
    assert.equal(f.run(`cameras.get('${camera}').overlay.children.length`), 0);
  }
  f.select(''); assert.equal(f.hidden(), true);
  assert.deepEqual(f.sent, []);
});

test('choice survives configure/reconnect; unavailable frames/aspect fail closed', () => {
  const f = fixture(); f.run(f.config); f.run(f.frames); f.select(catalog.templates[0].id);
  f.run('socket.onclose()'); assert.equal(f.hidden(), true);
  f.run(f.config); assert.equal(f.hidden(), true);
  f.run(f.frames); assert.equal(f.hidden(), false);
  assert.equal(f.run(`cameras.get('front_img_1').overlay.children.length`), 6);
  f.run(`cameras.get('front_img_1').image.naturalWidth = 1280; cameras.get('front_img_1').image.onload()`);
  assert.equal(f.hidden(), true);
  f.run(`cameras.get('front_img_1').image.naturalHeight = 960; cameras.get('front_img_1').image.onload()`);
  assert.equal(f.hidden(), false); // same aspect, different resolution
  f.run(`frame({status:'disarmed',age_ms:0,images:{}})`); assert.equal(f.hidden(), true);
  f.run(f.frames); f.run(`cameras.get('front_img_1').image.onerror()`); assert.equal(f.hidden(), true);
  assert.deepEqual(f.sent, []);
});

test('unknown camera, unknown choice, malformed catalog do not affect teleop', () => {
  const f = fixture(); f.run(f.config); f.run(f.frames); f.select('missing'); assert.equal(f.hidden(), true);
  f.select(catalog.templates[0].id); f.run(`configure({cameras:['left_wrist_img'],keys:{home:'y'}})`);
  assert.match(f.document.getElementById('reset-guide-status').textContent, /Waiting for front_img_1/);
  const broken = fixture({version: 1, templates: [{id:'broken'}]});
  assert.equal(broken.document.getElementById('reset-guide').disabled, true);
  broken.run(broken.config); broken.run(broken.frames);
  assert.match(broken.document.getElementById('reset-guide-status').textContent, /unavailable/);
  assert.deepEqual(broken.sent, []);
});

test('keyboard interactions on template select cannot arm or record', () => {
  const f = fixture(); f.run(f.config); f.run(f.frames);
  for (const key of ['g', 'b', 'y', ' ']) {
    f.document.onkeydown({key, code: key === ' ' ? 'Space' : '', target: f.document.getElementById('reset-guide')});
  }
  assert.deepEqual(f.sent, []);
});
