const $ = id => document.getElementById(id);
const SUPERSEDED = 4001;   // matches rollout_dashboard.SUPERSEDED_CLOSE_CODE
const cameras = new Map();
let socket;
let expectedImages = [];
let retryDelay = 500;
let overlayCamera;
let paused = false;
let started = false;
let waitForStart = false;
let videoEnabled = false;
let videoRecording = false;
let selectedVideo;
let modelBrowserEnabled = false;
let currentCheckpoint;
let modelDirectory = '.';
let modelLoadRevision = 0;
let modelSearchTimer;
let inferenceControls = {};
let inferenceControlSignature = '';
let inferenceControlRevision = 0;
let inferenceDrafts = {};
let pendingInferenceApply;
let inferenceApplyFeedback = '';

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

function reconnectCameras() {
  if (!confirm('Reconnect all configured RGB cameras? Rollout control will pause and require c to resume.')) return;
  if (!send({reconnect_cameras: true})) {
    reportDisconnected();
    return;
  }
  $('status').textContent = 'Reconnecting RGB cameras; replug the camera, then wait for Ready.';
  $('status').className = 'starting';
}

function updatePauseButton() {
  $('pause').disabled = !started;
  $('pause').textContent = paused ? 'Resume rollout (Space)' : 'Pause rollout (Space)';
}

function updateVideoControls() {
  $('recording-indicator').hidden = !videoRecording;
  $('record-video').disabled = !videoEnabled || (!started && !videoRecording);
  $('record-video').textContent = videoRecording ? 'Save video (v)' : 'Record video (v)';
  $('open-videos').disabled = !videoEnabled;
}

function updateModelControl() {
  $('select-model').disabled = !modelBrowserEnabled;
}

function updateCurrentModel() {
  const text = currentCheckpoint
    ? `Selected model: ${currentCheckpoint}`
    : 'Selected model: unavailable';
  $('current-model').textContent = text;
  $('current-model').title = text;
}

async function loadModels(path = '.', query = $('model-search').value) {
  if (!modelBrowserEnabled) return;
  const revision = ++modelLoadRevision;
  const list = $('model-list');
  list.textContent = 'Loading…';
  try {
    const parameters = new URLSearchParams({path, query});
    const response = await fetch(`/api/checkpoints?${parameters}`, {cache: 'no-store'});
    if (!response.ok) throw Error(`Could not load checkpoint directory (${response.status})`);
    const listing = await response.json();
    if (revision !== modelLoadRevision) return;
    modelDirectory = listing.path;
    $('model-path').textContent = `Folder: ${listing.path}`;
    $('model-up').disabled = listing.parent === null;
    list.replaceChildren();
    for (const entry of listing.entries) {
      const row = document.createElement('button');
      row.className = `model-row ${entry.type}`;
      const kind = document.createElement('span');
      kind.className = 'model-kind';
      kind.textContent = entry.type === 'directory' ? 'Folder' : 'Checkpoint';
      const name = document.createElement('span');
      name.className = 'model-name';
      name.textContent = entry.name;
      row.append(kind, name);
      if (entry.type === 'directory') {
        row.onclick = () => {
          $('model-search').value = '';
          loadModels(entry.path, '');
        };
      } else {
        row.onclick = () => selectModel(entry);
      }
      list.append(row);
    }
    if (!listing.entries.length) {
      list.textContent = query.trim()
        ? 'No folders or .ckpt files contain that text.'
        : 'No folders or .ckpt files here.';
    }
    $('model-up').onclick = () => {
      if (listing.parent === null) return;
      $('model-search').value = '';
      loadModels(listing.parent, '');
    };
  } catch (error) {
    if (revision !== modelLoadRevision) return;
    list.textContent = error.message;
  }
}

function selectModel(entry) {
  if (!confirm(`Load ${entry.name}? Existing plan actions will be discarded, the robot will hold position, and you must press Start after loading.`)) return;
  if (!send({select_model: entry.path})) {
    reportDisconnected();
    return;
  }
  $('models').close();
  $('select-model').disabled = true;
  $('status').textContent = `Loading ${entry.name}; rollout control is paused.`;
  $('status').className = 'starting';
}

function toggleVideoRecording() {
  if (!videoEnabled || (!started && !videoRecording)) return;
  if (!send({record_video: true})) {
    reportDisconnected();
    return;
  }
  $('record-video').disabled = true;
  $('record-video').textContent = videoRecording ? 'Saving video…' : 'Starting video…';
}

function togglePause() {
  if (!send({paused: !paused})) reportDisconnected();
}

function parsedInferenceDraft(name) {
  const spec = inferenceControls[name];
  const raw = inferenceDrafts[name] ?? '';
  const value = Number(raw);
  const valid = Boolean(spec) && raw.trim() !== '' && Number.isInteger(value)
    && value >= spec.min && value <= spec.max
    && (value - spec.min) % spec.step === 0;
  return {valid, value, dirty: valid && value !== spec?.value};
}

function updateInferenceApplyState() {
  const button = $('apply-inference');
  const status = $('inference-apply-status');
  if (!button || !status) return;
  const parsed = Object.keys(inferenceControls).map(parsedInferenceDraft);
  const invalid = parsed.some(item => !item.valid);
  const dirty = parsed.some(item => item.dirty);
  button.disabled = Boolean(pendingInferenceApply) || invalid || !dirty
    || socket?.readyState !== WebSocket.OPEN;
  button.textContent = pendingInferenceApply ? 'Queued…' : 'Apply settings';
  if (pendingInferenceApply) {
    status.textContent = started && !paused
      ? 'Queued; the current action prefix will finish before these settings apply.'
      : 'Applying settings before the next plan.';
  } else if (invalid) {
    status.textContent = 'Enter values within the limits declared by this model profile.';
  } else if (dirty) {
    status.textContent = 'Unapplied changes. Press Enter or Apply settings.';
  } else {
    status.textContent = inferenceApplyFeedback
      || 'Active values from the selected model profile.';
  }
}

function editInferenceControl(name, input) {
  inferenceDrafts[name] = input.value;
  inferenceApplyFeedback = '';
  const parsed = parsedInferenceDraft(name);
  input.setCustomValidity(parsed.valid ? '' : 'Use a valid value in the declared range.');
  input.dataset.dirty = String(parsed.dirty);
  updateInferenceApplyState();
}

function applyInferenceOverrides() {
  if (pendingInferenceApply) return;
  const values = {};
  for (const name of Object.keys(inferenceControls)) {
    const parsed = parsedInferenceDraft(name);
    if (!parsed.valid) {
      $(`inference-${name}`)?.reportValidity();
      return;
    }
    if (parsed.dirty) values[name] = parsed.value;
  }
  if (!Object.keys(values).length) return;
  if (!send({inference_override: values})) {
    reportDisconnected();
    return;
  }
  pendingInferenceApply = {values, afterRevision: inferenceControlRevision};
  setInferenceControlsDisabled(true);
  updateInferenceApplyState();
}

function setInferenceControlsDisabled(disabled) {
  for (const input of $('inference-controls').querySelectorAll('input')) {
    input.disabled = disabled;
  }
  const button = $('apply-inference');
  if (button) button.disabled = disabled;
  if (!disabled) updateInferenceApplyState();
}

function updateInferenceControls(controls, revision = inferenceControlRevision) {
  inferenceControls = controls && typeof controls === 'object' ? controls : {};
  const entries = Object.entries(inferenceControls);
  const signature = JSON.stringify(entries.map(([name, spec]) => [
    name, spec.label, spec.description, spec.type, spec.min, spec.max, spec.step,
  ]));
  const container = $('inference-controls');
  if (signature !== inferenceControlSignature) {
    inferenceControlSignature = signature;
    inferenceDrafts = {};
    pendingInferenceApply = undefined;
    inferenceApplyFeedback = '';
    container.replaceChildren();
    for (const [name, spec] of entries) {
      const label = document.createElement('label');
      label.className = 'inference-control';
      label.title = spec.description || spec.label;
      label.htmlFor = `inference-${name}`;
      label.append(document.createTextNode(spec.label));
      const input = document.createElement('input');
      input.id = `inference-${name}`;
      input.type = 'number';
      input.min = String(spec.min);
      input.max = String(spec.max);
      input.step = String(spec.step);
      input.value = String(spec.value);
      input.defaultValue = String(spec.value);
      inferenceDrafts[name] = input.value;
      input.oninput = event => editInferenceControl(name, event.target);
      input.onkeydown = event => {
        if (event.key === 'Enter') {
          event.preventDefault();
          applyInferenceOverrides();
        }
      };
      label.append(input);
      container.append(label);
    }
    const apply = document.createElement('button');
    apply.id = 'apply-inference';
    apply.className = 'inference-apply';
    apply.type = 'button';
    apply.onclick = applyInferenceOverrides;
    apply.textContent = 'Apply settings';
    const status = document.createElement('span');
    status.id = 'inference-apply-status';
    status.className = 'inference-apply-status';
    status.setAttribute('aria-live', 'polite');
    container.append(apply, status);
  }
  container.hidden = entries.length === 0;
  const hasRevision = Number.isInteger(revision);
  const nextRevision = hasRevision ? revision : inferenceControlRevision;
  const pendingValuesApplied = pendingInferenceApply
    && Object.entries(pendingInferenceApply.values).every(
      ([name, value]) => inferenceControls[name]?.value === value,
    );
  if (pendingInferenceApply && (
    nextRevision > pendingInferenceApply.afterRevision
    || (!hasRevision && pendingValuesApplied)
  )) {
    inferenceApplyFeedback = pendingValuesApplied
      ? 'Applied atomically at the replan boundary.'
      : 'Settings were rejected; active model values were restored.';
    pendingInferenceApply = undefined;
    for (const [name, spec] of entries) inferenceDrafts[name] = String(spec.value);
  }
  inferenceControlRevision = nextRevision;
  for (const [name, spec] of entries) {
    const input = $(`inference-${name}`);
    const parsed = parsedInferenceDraft(name);
    if (input && !pendingInferenceApply && !parsed.dirty && document.activeElement !== input) {
      inferenceDrafts[name] = String(spec.value);
      input.value = String(spec.value);
      input.defaultValue = String(spec.value);
      input.dataset.dirty = 'false';
      input.setCustomValidity('');
    }
  }
  setInferenceControlsDisabled(Boolean(pendingInferenceApply));
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
  videoEnabled = Boolean(message.video_recording_enabled);
  videoRecording = Boolean(message.video_recording);
  modelBrowserEnabled = Boolean(message.model_browser_enabled);
  currentCheckpoint = message.checkpoint;
  updateCurrentModel();
  updateInferenceControls(
    message.inference_controls, message.inference_controls_revision,
  );
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
    // A canvas, not an <img>: each rollout frame is drawn and then released.
    // Assigning a fresh data: URL per frame would leave one document resource
    // per camera per frame for the browser to hold for the whole rollout.
    const view = document.createElement('canvas');
    view.setAttribute('role', 'img');
    view.setAttribute('aria-label', `${name} live RGB`);
    head.append(title, detail);
    card.append(head, view);
    $('cameras').append(card);
    cameras.set(name, {card, detail, view, context: view.getContext('2d'), busy: false});
  }
  $('overlay').checked = Boolean(message.overlay_enabled);
  $('overlay').disabled = false;
  $('start').disabled = !waitForStart || started;
  updatePauseButton();
  updateVideoControls();
  updateModelControl();
  $('restart').disabled = false;
  $('reconnect-cameras').disabled = false;
}

function frame(message) {
  paused = Boolean(message.paused);
  started = Boolean(message.started);
  videoRecording = Boolean(message.video_recording);
  currentCheckpoint = message.checkpoint;
  updateCurrentModel();
  updateInferenceControls(
    message.inference_controls, message.inference_controls_revision,
  );
  $('start').disabled = !waitForStart || started;
  updatePauseButton();
  updateVideoControls();
  updateModelControl();
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
    if (message.images.includes(name)) {
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

async function drawCameraFrame(name, data) {
  const tile = cameras.get(name);
  // Drop the frame while the previous one for this camera is still decoding.
  // The next frame is 80 ms away; a backlog would never drain.
  if (!tile || tile.busy) return;
  tile.busy = true;
  try {
    const bitmap = await createImageBitmap(new Blob([data], {type: 'image/jpeg'}));
    if (tile.view.width !== bitmap.width || tile.view.height !== bitmap.height) {
      tile.view.width = bitmap.width;
      tile.view.height = bitmap.height;
    }
    tile.context.drawImage(bitmap, 0, 0);
    bitmap.close();
  } catch {
    // A truncated frame is discarded; the next one repaints the camera.
  } finally {
    tile.busy = false;
  }
}

function formatDuration(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds - minutes * 60);
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
}

function markSelectedVideo() {
  for (const row of document.querySelectorAll('.video-row')) {
    row.classList.toggle('selected', row.dataset.id === selectedVideo?.id);
  }
}

function openVideo(video) {
  selectedVideo = video;
  $('video-empty').hidden = true;
  $('video-player-wrap').hidden = false;
  $('video-title').textContent = video.id;
  $('video-meta').textContent = `${video.frames} frames · ${video.fps} fps · ${formatDuration(video.duration_seconds)} · synchronized three-camera mosaic`;
  const player = $('video-player');
  player.pause();
  player.src = `/api/videos/${encodeURIComponent(video.id)}`;
  player.load();
  markSelectedVideo();
}

async function loadVideos() {
  const list = $('video-list');
  list.textContent = 'Loading…';
  try {
    const response = await fetch('/api/videos', {cache: 'no-store'});
    if (!response.ok) throw Error(`Could not load rollout videos (${response.status})`);
    const videos = await response.json();
    list.replaceChildren();
    for (const video of videos) {
      const row = document.createElement('button');
      row.className = 'video-row';
      row.dataset.id = video.id;
      const name = document.createElement('span');
      name.className = 'video-name';
      name.textContent = video.id;
      const detail = document.createElement('span');
      detail.className = 'video-detail';
      detail.textContent = `${formatDuration(video.duration_seconds)} · ${video.frames} frames`;
      row.append(name, detail);
      row.onclick = () => openVideo(video);
      list.append(row);
    }
    if (!videos.length) {
      selectedVideo = undefined;
      $('video-player').pause();
      $('video-player').removeAttribute('src');
      $('video-player').load();
      $('video-player-wrap').hidden = true;
      $('video-empty').hidden = false;
      list.textContent = 'No completed rollout videos yet.';
      return;
    }
    const current = videos.find(video => video.id === selectedVideo?.id);
    if (current) {
      selectedVideo = current;
      markSelectedVideo();
    } else {
      openVideo(videos[0]);
    }
  } catch (error) {
    list.textContent = error.message;
  }
}

$('stop').onclick = stopRollout;
$('start').onclick = startRollout;
$('pause').onclick = togglePause;
$('record-video').onclick = toggleVideoRecording;
$('select-model').onclick = () => {
  if (!modelBrowserEnabled) return;
  $('model-current').textContent = currentCheckpoint ? `Current: ${currentCheckpoint}` : 'Current checkpoint unavailable';
  $('model-search').value = '';
  $('models').showModal();
  loadModels(modelDirectory, '');
  $('model-search').focus();
};
$('open-videos').onclick = () => {
  $('videos').showModal();
  loadVideos();
};
$('refresh-videos').onclick = loadVideos;
$('close-videos').onclick = () => {
  $('video-player').pause();
  $('videos').close();
};
$('model-refresh').onclick = () => loadModels(modelDirectory);
$('model-search').oninput = () => {
  clearTimeout(modelSearchTimer);
  modelSearchTimer = setTimeout(() => loadModels(modelDirectory), 120);
};
$('model-close').onclick = () => $('models').close();
$('restart').onclick = restartRollout;
$('reconnect-cameras').onclick = reconnectCameras;
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
    || event.target instanceof HTMLVideoElement
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
  if (event.key === 'v' || event.key === 'V') {
    event.preventDefault();
    toggleVideoRecording();
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
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    retryDelay = 500;
    expectedImages = [];
    $('status').textContent = 'Connected; waiting for rollout frames';
    updateInferenceApplyState();
  };
  socket.onmessage = event => {
    // Camera JPEGs arrive as binary messages, in the order the preceding frame
    // status named them.
    if (typeof event.data !== 'string') {
      const name = expectedImages.shift();
      if (name) drawCameraFrame(name, event.data);
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === 'config') configure(message);
    if (message.type === 'frame') {
      expectedImages = message.images.slice();
      frame(message);
    }
  };
  socket.onclose = event => {
    $('status').className = 'starting';
    if (event.code === SUPERSEDED) {
      // A newer tab owns the dashboard. Reconnecting would only take it back
      // and leave the two tabs fighting over the rollout.
      $('status').textContent = 'A newer tab took over this dashboard; close this one.';
      for (const id of ['start', 'pause', 'record-video', 'open-videos', 'select-model',
                        'restart', 'reconnect-cameras', 'overlay']) $(id).disabled = true;
      setInferenceControlsDisabled(true);
      return;
    }
    $('status').textContent = 'Dashboard disconnected; retrying…';
    $('start').disabled = true;
    $('pause').disabled = true;
    $('record-video').disabled = true;
    $('open-videos').disabled = true;
    $('select-model').disabled = true;
    setInferenceControlsDisabled(true);
    $('restart').disabled = true;
    $('reconnect-cameras').disabled = true;
    $('overlay').disabled = true;
    // Back off: a tab left open from a finished rollout should not poll the
    // station once a second for the rest of the day.
    setTimeout(connect, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 10000);
  };
}

connect();
