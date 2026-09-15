const $ = id => document.getElementById(id);
const cameras = new Map();
let socket;
let overlayCamera;
let paused = false;
let started = false;
let waitForStart = false;

function send(message) {
  if (socket?.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(message));
  return true;
}

function reportDisconnected() {
  $('status').textContent = 'Dashboard is not connected yet; wait for Ready, then press c.';
  $('status').className = 'starting';
}

function stopRollout() {
  if (confirm('Stop this rollout? The robot will follow the existing rollout shutdown path.')) {
    if (!send({stop: true})) {
      reportDisconnected();
      return;
    }
    $('stop').disabled = true;
    $('stop').textContent = 'Stopping rollout…';
  }
}

function startRollout() {
  if (!send({start: true})) {
    reportDisconnected();
    return;
  }
  $('start').disabled = true;
  $('start').textContent = 'Starting…';
}

function restartRollout() {
  if (!send({restart: true})) {
    reportDisconnected();
    return false;
  }
  $('start').disabled = false;
  $('start').textContent = 'Start rollout (c)';
  return true;
}

function updatePauseButton() {
  $('pause').disabled = !started;
  $('pause').textContent = paused ? 'Resume rollout (Space)' : 'Pause rollout (Space)';
}

function togglePause() {
  if (!send({paused: !paused})) {
    reportDisconnected();
  }
}

function setExecuteSteps(value) {
  const executeSteps = Number(value);
  if (!Number.isInteger(executeSteps) || executeSteps < 1 || executeSteps > 100) {
    $('execute-steps').value = String($('execute-steps').defaultValue);
    return;
  }
  if (!send({execute_steps: executeSteps})) {
    reportDisconnected();
  }
}

function updateInference(inference) {
  if (!inference || !inference.samples) {
    $('inference-status').textContent = 'Inference: waiting for first plan';
    return;
  }
  $('inference-status').textContent = `Inference: last ${inference.last_ms.toFixed(0)} ms · rolling average ${inference.mean_ms.toFixed(0)} ms · ${inference.plans_per_second.toFixed(2)} plans/s`;
}

function chooseVelocityAction(action) {
  if (action === 'restart') {
    if (restartRollout()) $('velocity-decision').hidden = true;
    return;
  }
  if (send({velocity_action: action})) $('velocity-decision').hidden = true;
}

function configure(message) {
  waitForStart = Boolean(message.wait_for_start);
  paused = Boolean(message.paused);
  started = Boolean(message.started);
  $('execute-steps').value = String(message.execute_steps);
  $('execute-steps').defaultValue = String(message.execute_steps);
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
  $('execute-steps').disabled = false;
  $('start').disabled = !waitForStart || started;
  updatePauseButton();
  $('restart').disabled = false;
}

function frame(message) {
  paused = Boolean(message.paused);
  started = Boolean(message.started);
  if (document.activeElement !== $('execute-steps')) {
    $('execute-steps').value = String(message.execute_steps);
    $('execute-steps').defaultValue = String(message.execute_steps);
  }
  $('start').disabled = !waitForStart || started;
  updatePauseButton();
  $('status').textContent = `${message.status} · camera update ${message.age_ms} ms ago`;
  $('status').className = message.status === 'Running' ? 'running' : 'starting';
  $('overlay-status').textContent = message.overlay_enabled
    ? `${message.overlay_status} · showing on ${overlayCamera}`
    : `${message.overlay_status} · overlay hidden`;
  updateInference(message.inference);
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
  const prompt = message.velocity_prompt;
  $('velocity-decision').hidden = !prompt;
  if (prompt) {
    $('velocity-detail').textContent = `${prompt.arms.join(' and ')} target step ${prompt.max_joint_step.toFixed(3)} rad exceeds ${prompt.limit.toFixed(3)} rad.`;
  }
}

$('stop').onclick = stopRollout;
$('start').onclick = startRollout;
$('pause').onclick = togglePause;
$('execute-steps').onchange = event => setExecuteSteps(event.target.value);
$('restart').onclick = restartRollout;
for (const button of document.querySelectorAll('[data-velocity-action]')) {
  button.onclick = () => chooseVelocityAction(button.dataset.velocityAction);
}
$('overlay').onchange = event => send({overlay: event.target.checked});
document.onkeydown = event => {
  if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
  if (
    event.target instanceof HTMLInputElement
    || event.target instanceof HTMLButtonElement
    || event.target instanceof HTMLSelectElement
    || event.target instanceof HTMLTextAreaElement
  ) return;
  if (event.code === 'Space') {
    event.preventDefault();
    togglePause();
    return;
  }
  if (event.key === 'q' || event.key === 'Q' || event.key === 'Escape') {
    event.preventDefault();
    stopRollout();
  }
  if (event.key === 'c' || event.key === 'C') {
    event.preventDefault();
    startRollout();
  }
  if (event.key === 'r' || event.key === 'R') {
    event.preventDefault();
    restartRollout();
  }
  if (event.key === 'e' || event.key === 'E') {
    event.preventDefault();
    chooseVelocityAction('execute');
  }
  if (event.key === 's' || event.key === 'S') {
    event.preventDefault();
    chooseVelocityAction('resample');
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
    $('start').disabled = true;
    $('pause').disabled = true;
    $('execute-steps').disabled = true;
    $('restart').disabled = true;
    $('overlay').disabled = true;
    setTimeout(connect, 1000);
  };
}

connect();
