const cameras = new Map();
const $ = id => document.getElementById(id);
let socket;
let currentEpisode;
let pendingEpisode;
let pendingEpisodeFrames = 0;
let pendingEpisodeLength;
let episodeLengthDirty = false;
let episodeLengthRequestedAt = 0;
let review;
let timer;
const playbackFps = 30;

const labels = {
  activate_both: 'Arm both followers', record: 'Start / save recording',
  stop: 'Disarm followers', quit: 'Quit teleop',
};
const keyLabel = key => key === ' ' ? 'Space' : key;

function send(key) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({key}));
}

function sendGripperForce(arm, value) {
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({gripper_force: {arm, value: Number(value)}}));
  }
}

function reconnectCameras() {
  if (socket?.readyState !== WebSocket.OPEN) {
    $('status').textContent = 'Dashboard is not connected.';
    return;
  }
  if (!confirm('Reconnect all RGB cameras? Followers will disarm and teleop must be armed again.')) return;
  $('reconnect-cameras').disabled = true;
  $('status').textContent = 'Camera reconnect requested…';
  socket.send(JSON.stringify({reconnect_cameras: true}));
}

function configure(message) {
  cameras.clear();
  $('cameras').replaceChildren();
  $('controls').replaceChildren();
  const gripperRoot = $('gripper-controls');
  gripperRoot.replaceChildren();
  const forceLimits = message.gripper_force_limits ?? {};
  for (const [arm, value] of Object.entries(forceLimits)) {
    const label = document.createElement('label');
    label.className = 'gripper-control';
    const name = document.createElement('span');
    name.textContent = `${arm} arm`;
    const slider = document.createElement('input');
    slider.type = 'range'; slider.min = '1'; slider.max = '50'; slider.step = '1'; slider.value = value;
    const readout = document.createElement('output');
    readout.textContent = `${value} N`;
    slider.oninput = () => { readout.textContent = `${slider.value} N`; };
    slider.onchange = () => sendGripperForce(arm, slider.value);
    label.append(name, slider, readout);
    gripperRoot.append(label);
  }
  $('gripper-settings').hidden = Object.keys(forceLimits).length === 0;
  for (const name of message.cameras) {
    const card = document.createElement('article');
    card.className = 'camera waiting';
    const head = document.createElement('div');
    head.className = 'camera-head';
    const title = document.createElement('strong');
    title.textContent = name.replaceAll('_', ' ');
    const age = document.createElement('small');
    age.textContent = 'Waiting for frame';
    const image = document.createElement('img');
    image.alt = `${name} live RGB`;
    image.draggable = false;
    head.append(title, age);
    card.append(head, image);
    $('cameras').append(card);
    cameras.set(name, {card, age, image});
  }
  const reverse = Object.fromEntries(
    Object.entries(message.keys).map(([action, key]) => [key, action]),
  );
  for (const [key, action] of Object.entries(reverse)) {
    if (action === 'home') continue;
    const button = document.createElement('button');
    button.textContent = `${labels[action] ?? action} [${keyLabel(key)}]`;
    button.onclick = () => send(key);
    $('controls').append(button);
  }
  $('reset-home').onclick = () => {
    if (confirm('Reset both YAM arms to home?')) send(message.keys.home);
  };
  $('reconnect-cameras').disabled = false;
  $('reconnect-cameras').onclick = reconnectCameras;
  updateEpisodeLength(message);
  const isEditable = target => target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement
    || (target instanceof HTMLElement && target.isContentEditable);
  document.onkeydown = event => {
    if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) return;
    if (isEditable(event.target)) return;
    const key = event.code === 'Space' ? ' ' : event.key.toLowerCase();
    if (reverse[key]) {
      event.preventDefault();
      send(key);
    }
  };
}

function updateEpisode(message) {
  if (!Number.isInteger(message.episode)) return;
  const input = $('episode-number');
  const locked = message.episode_state === 'recording';
  const queued = message.episode_state === 'queued';
  currentEpisode = message.episode;
  input.disabled = locked;
  $('set-episode').disabled = locked;
  const stateLabels = {
    next: 'Next episode ID', queued: 'Queued episode ID', recording: 'Recording episode ID',
  };
  $('episode-label').textContent = stateLabels[message.episode_state] ?? 'Episode ID';
  if (pendingEpisode !== undefined) {
    if (pendingEpisode === currentEpisode) {
      input.value = currentEpisode;
      $('episode-hint').textContent = `Using demo_${currentEpisode}.hdf5`;
      pendingEpisode = undefined;
      pendingEpisodeFrames = 0;
    } else if (locked) {
      input.value = currentEpisode;
      $('episode-hint').textContent = 'ID cannot change after recording has started.';
      pendingEpisode = undefined;
      pendingEpisodeFrames = 0;
    } else if (++pendingEpisodeFrames >= 6) {
      const rejected = pendingEpisode;
      input.value = rejected;
      $('episode-hint').textContent = `ID ${rejected} was not accepted; choose an unused ID and try again.`;
    } else {
      input.value = pendingEpisode;
      $('episode-hint').textContent = `Applying episode ${pendingEpisode}…`;
    }
  } else if (locked) {
    input.value = currentEpisode;
    $('episode-hint').textContent = 'ID cannot change after recording has started.';
  } else {
    if (document.activeElement !== input) input.value = currentEpisode;
    $('episode-hint').textContent = queued
      ? `Queued as demo_${currentEpisode}.hdf5; editable until recording starts.`
      : `Will save as demo_${currentEpisode}.hdf5`;
  }
}

function frame(message) {
  $('status').textContent = `${message.status} · camera update ${message.age_ms} ms ago`;
  $('status').className = message.status.startsWith('active')
    ? 'active' : message.status.startsWith('armed') ? 'armed' : 'disarmed';
  $('recording-indicator').hidden = !message.recording;
  const recovering = message.status.startsWith('Camera reconnect')
    || message.status.startsWith('Reconnecting RGB');
  $('reconnect-cameras').disabled = recovering;
  updateEpisode(message);
  updateEpisodeLength(message);
  for (const [arm, value] of Object.entries(message.gripper_force_limits ?? {})) {
    const slider = [...document.querySelectorAll('.gripper-control')].find(el => el.firstChild?.textContent === `${arm} arm`)
      ?.querySelector('input');
    if (slider && document.activeElement !== slider) {
      slider.value = value;
      slider.nextElementSibling.textContent = `${value} N`;
    }
  }
  for (const [name, tile] of cameras) {
    const image = message.images[name];
    if (image) {
      tile.image.src = image;
      tile.age.textContent = message.recording ? 'RECORDING' : 'LIVE';
      tile.card.classList.remove('waiting');
    } else {
      tile.age.textContent = 'Waiting for camera';
      tile.card.classList.add('waiting');
    }
  }
}

function updateEpisodeLength(message) {
  if (!Object.hasOwn(message, 'episode_length')) return;
  const limit = message.episode_length;
  const enabled = $('episode-limit-enabled');
  const input = $('episode-length');
  if (enabled.disabled) {
    enabled.disabled = false;
    $('set-episode-length').disabled = false;
    $('episode-length-hint').textContent = 'Off means unlimited. Changes apply to the current recording.';
  }
  if (pendingEpisodeLength !== undefined) {
    if (pendingEpisodeLength === limit) {
      pendingEpisodeLength = undefined;
      episodeLengthDirty = false;
      $('episode-length-hint').textContent = limit === null
        ? 'Limit off. Save the recording manually.'
        : 'Applied. Lowering the limit below the current count saves the take.';
    } else if (Date.now() - episodeLengthRequestedAt > 3000) {
      pendingEpisodeLength = undefined;
      $('episode-length-hint').textContent = 'Change was not accepted. Press Apply to retry.';
    }
  }
  if (!episodeLengthDirty && document.activeElement !== input) {
    enabled.checked = limit !== null;
    if (limit !== null) input.value = limit;
  }
  input.disabled = !enabled.checked;
  const count = message.recorded_frames ?? 0;
  const rate = message.recording_rate_hz ?? playbackFps;
  $('episode-length-status').textContent = limit === null
    ? `Unlimited · ${count.toLocaleString()} frames recorded`
    : `${count.toLocaleString()} / ${limit.toLocaleString()} frames · auto-save at ${(limit / rate).toFixed(1)} s`;
}

function submitEpisodeLength(event) {
  event.preventDefault();
  if ($('episode-limit-enabled').disabled) return;
  const enabled = $('episode-limit-enabled').checked;
  const input = $('episode-length');
  const hint = $('episode-length-hint');
  const limit = enabled ? Number(input.value) : null;
  input.disabled = !enabled;
  episodeLengthDirty = true;
  if (enabled && (!/^\d+$/.test(input.value.trim()) || !Number.isSafeInteger(limit) || limit < 1)) {
    hint.textContent = 'Enter a positive whole number of frames.';
    return;
  }
  if (socket?.readyState !== WebSocket.OPEN) {
    hint.textContent = 'Dashboard is not connected. Reconnect, then press Apply.';
    return;
  }
  pendingEpisodeLength = limit;
  episodeLengthRequestedAt = Date.now();
  hint.textContent = 'Applying episode limit…';
  socket.send(JSON.stringify({episode_length: limit}));
}

async function loadDemos() {
  const list = $('demo-list');
  list.textContent = 'Loading…';
  try {
    const response = await fetch('/api/demos', {cache: 'no-store'});
    if (!response.ok) throw Error(`Could not load demos (${response.status})`);
    const demos = await response.json();
    list.replaceChildren();
    for (const demo of demos) {
      const row = document.createElement('div');
      row.className = `demo-row${review?.id === demo.id ? ' selected' : ''}`;
      row.dataset.id = demo.id;
      const select = document.createElement('button');
      select.className = 'demo-select';
      const name = document.createElement('span');
      name.className = 'demo-name';
      name.textContent = `demo_${demo.id}`;
      const frames = document.createElement('span');
      frames.className = 'demo-frames';
      frames.textContent = `${demo.frames} frames · ${(demo.frames / playbackFps).toFixed(1)} s`;
      select.append(name, frames);
      select.onclick = () => openDemo(demo);
      const remove = document.createElement('button');
      remove.className = 'demo-delete';
      remove.textContent = 'Delete';
      remove.title = `Delete demo_${demo.id}.hdf5`;
      remove.onclick = () => deleteDemo(demo);
      row.append(select, remove);
      list.append(row);
    }
    if (!demos.length) {
      review = undefined;
      $('player').hidden = true;
      $('demo-empty').hidden = false;
      list.textContent = 'No completed demos yet.';
    } else if (!review || !demos.some(demo => demo.id === review.id)) {
      openDemo(demos[0]);
    }
  } catch (error) {
    list.textContent = error.message;
  }
}

async function showFrame(index) {
  if (!review) return;
  const response = await fetch(`/api/demos/${review.id}/frame?index=${index}`);
  const data = await response.json();
  if (!response.ok || data.error) throw Error(data.error ?? `Frame request failed (${response.status})`);
  $('seek').max = data.frames - 1;
  $('seek').value = data.index;
  $('frame').textContent = `Frame ${data.index + 1} of ${data.frames}`;
  $('playback-time').textContent = `${formatTime(data.index / playbackFps)} / ${formatTime((data.frames - 1) / playbackFps)}`;
  const root = $('review-cameras');
  if (!root.children.length) {
    for (const name of Object.keys(data.images)) {
      const card = document.createElement('article');
      card.className = 'review-camera';
      const title = document.createElement('strong');
      title.textContent = name.replaceAll('_', ' ');
      const img = document.createElement('img');
      img.alt = `${name} recorded RGB`;
      img.dataset.name = name;
      card.append(title, img);
      root.append(card);
    }
  }
  for (const img of root.querySelectorAll('img')) img.src = data.images[img.dataset.name];
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return `${String(minutes).padStart(2, '0')}:${remainder.toFixed(2).padStart(5, '0')}`;
}

function openDemo(demo) {
  review = demo;
  $('review-title').textContent = `demo_${demo.id}.hdf5`;
  $('demo-empty').hidden = true;
  $('player').hidden = false;
  $('review-cameras').replaceChildren();
  clearInterval(timer);
  timer = undefined;
  $('play').textContent = '▶ Play';
  $('frame').textContent = 'Loading first frame…';
  for (const row of document.querySelectorAll('.demo-row')) {
    row.classList.toggle('selected', Number(row.dataset.id) === demo.id);
  }
  showFrame(0).catch(error => { $('frame').textContent = error.message; });
}

async function deleteDemo(demo) {
  if (!confirm(`Permanently delete demo_${demo.id}.hdf5?`)) return;
  try {
    const response = await fetch(`/api/demos/${demo.id}`, {method: 'DELETE'});
    const data = await response.json();
    if (!response.ok || data.error) throw Error(data.error ?? `Delete failed (${response.status})`);
    if (review?.id === demo.id) {
      review = undefined;
      clearInterval(timer);
      timer = undefined;
      $('player').hidden = true;
      $('demo-empty').hidden = false;
      $('review-cameras').replaceChildren();
    }
    await loadDemos();
  } catch (error) {
    alert(error.message);
  }
}

function pausePlayback() {
  clearInterval(timer);
  timer = undefined;
  $('play').textContent = '▶ Play';
}

function stepPlayback(delta) {
  if (!review) return;
  pausePlayback();
  const index = Math.max(
    0,
    Math.min(Number($('seek').max), Number($('seek').value) + delta),
  );
  showFrame(index).catch(error => { $('frame').textContent = error.message; });
}

$('open-demos').onclick = () => {
  $('demos').showModal();
  loadDemos();
};
$('close-demos').onclick = () => {
  pausePlayback();
  $('demos').close();
};
function submitEpisode(event) {
  event.preventDefault();
  const input = $('episode-number');
  const raw = input.value.trim();
  const hint = $('episode-hint');
  if (!/^\d+$/.test(raw)) {
    hint.textContent = 'Enter a nonnegative whole number.';
    return;
  }
  const episode = Number(raw);
  if (!Number.isSafeInteger(episode)) {
    hint.textContent = 'Enter a smaller whole number.';
    return;
  }
  if (socket?.readyState !== WebSocket.OPEN) {
    hint.textContent = 'Dashboard is not connected.';
    return;
  }
  pendingEpisode = episode;
  pendingEpisodeFrames = 0;
  hint.textContent = `Requesting episode ${episode}…`;
  socket.send(JSON.stringify({episode}));
}
const episodeForm = $('episode-form');
episodeForm.onsubmit = submitEpisode;
episodeForm.onkeydown = event => event.stopPropagation();
$('episode-number').oninput = () => {
  pendingEpisode = undefined;
  pendingEpisodeFrames = 0;
};
const episodeLengthForm = $('episode-length-form');
episodeLengthForm.onsubmit = submitEpisodeLength;
episodeLengthForm.onkeydown = event => event.stopPropagation();
$('episode-limit-enabled').onchange = submitEpisodeLength;
$('episode-length').oninput = () => {
  episodeLengthDirty = true;
  pendingEpisodeLength = undefined;
  $('episode-length-hint').textContent = 'Press Apply to use this limit for the current recording.';
};
$('seek').oninput = () => {
  pausePlayback();
  showFrame(Number($('seek').value)).catch(error => { $('frame').textContent = error.message; });
};
$('previous-frame').onclick = () => stepPlayback(-1);
$('next-frame').onclick = () => stepPlayback(1);
$('play').onclick = () => {
  if (!review) return;
  if (timer) {
    pausePlayback();
    return;
  }
  $('play').textContent = '❚❚ Pause';
  timer = setInterval(() => {
    const next = Number($('seek').value) + 1;
    if (next > Number($('seek').max)) {
      pausePlayback();
    } else {
      showFrame(next).catch(error => {
        pausePlayback();
        $('frame').textContent = error.message;
      });
    }
  }, 1000 / playbackFps);
};

function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.onopen = () => { $('status').textContent = 'Connected; waiting for frames'; };
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.type === 'config') configure(message);
    if (message.type === 'frame') frame(message);
  };
  socket.onclose = () => {
    $('reconnect-cameras').disabled = true;
    $('status').textContent = 'Dashboard disconnected; retrying…';
    setTimeout(connect, 1000);
  };
}

connect();
