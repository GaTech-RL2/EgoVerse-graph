const $ = id => document.getElementById(id);
const cameras = new Map();
let socket;
let overlayCamera;

function send(message) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

function stopRollout() {
  if (confirm('Stop this rollout? The robot will follow the existing rollout shutdown path.')) {
    send({stop: true});
    $('stop').disabled = true;
    $('stop').textContent = 'Stopping rollout…';
  }
}

function configure(message) {
  overlayCamera = message.overlay_camera;
  cameras.clear();
  $('cameras').replaceChildren();
  for (const name of message.cameras) {
    const card = document.createElement('article');
    card.className = 'camera waiting';
    const head = document.createElement('div');
    head.className = 'camera-head';
    const title = document.createElement('strong');
    title.textContent = name.replaceAll('_', ' ');
    const detail = document.createElement('small');
    detail.textContent = name === overlayCamera ? 'Overlay-capable front view' : 'Live RGB';
    const image = document.createElement('img');
    image.alt = `${name} live RGB`;
    image.draggable = false;
    head.append(title, detail);
    card.append(head, image);
    $('cameras').append(card);
    cameras.set(name, {card, detail, image});
  }
  $('overlay').checked = Boolean(message.overlay_enabled);
  $('overlay').disabled = false;
}

function frame(message) {
  $('status').textContent = `${message.status} · camera update ${message.age_ms} ms ago`;
  $('status').className = message.status === 'Running' ? 'running' : 'starting';
  $('overlay-status').textContent = message.overlay_enabled
    ? `${message.overlay_status} · showing on ${overlayCamera}`
    : `${message.overlay_status} · overlay hidden`;
  if ($('overlay').checked !== Boolean(message.overlay_enabled)) {
    $('overlay').checked = Boolean(message.overlay_enabled);
  }
  for (const [name, tile] of cameras) {
    const image = message.images[name];
    if (image) {
      tile.image.src = image;
      tile.card.classList.remove('waiting');
      tile.detail.textContent = name === overlayCamera && message.overlay_enabled
        ? 'Cartesian action overlay' : 'Live RGB';
    } else {
      tile.card.classList.add('waiting');
      tile.detail.textContent = 'Waiting for camera';
    }
  }
}

$('stop').onclick = stopRollout;
$('overlay').onchange = event => send({overlay: event.target.checked});
document.onkeydown = event => {
  if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
  if (event.target instanceof HTMLInputElement) return;
  if (event.key === 'q' || event.key === 'Q' || event.key === 'Escape') {
    event.preventDefault();
    stopRollout();
  }
};

function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.onopen = () => { $('status').textContent = 'Connected; waiting for rollout frames'; };
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.type === 'config') configure(message);
    if (message.type === 'frame') frame(message);
  };
  socket.onclose = () => {
    $('status').textContent = 'Dashboard disconnected; retrying…';
    $('status').className = 'starting';
    setTimeout(connect, 1000);
  };
}

connect();
