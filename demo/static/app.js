const $ = (id) => document.getElementById(id);
const names = {bicubic: "Bicubic", realesrnet: "Real-ESRNet", realesrgan: "Real-ESRGAN", resshift: "ResShift"};
const methods = Object.keys(names);
const state = {catalog: null, current: null, selected: null, busy: false, reference: false};
const images = new Map();
let uploadFile = null;

async function api(path, data) {
  const response = await fetch(path, {
    method: data === undefined ? "GET" : "POST",
    headers: data === undefined ? {} : {"Content-Type": "application/json"},
    body: data === undefined ? undefined : JSON.stringify(data),
    signal: AbortSignal.timeout(120000),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "The request failed.");
  return body;
}

function error(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}

async function loadImage(url) {
  if (!url || images.has(url)) return;
  await new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => { images.set(url, image); resolve(); };
    image.onerror = () => reject(new Error("Could not load the image. Check that the demo server is running."));
    image.src = url;
  });
}

function available(method) {
  const current = state.current;
  if (!current) return false;
  if (method === "bicubic") return true;
  const live = state.catalog?.live_methods?.[method];
  return live?.enabled && Math.max(current.width, current.height) <= live.max_side
    && Math.min(current.width, current.height) >= (live.min_side || 16);
}

function setBusy(value) {
  state.busy = value;
  $("upload-button").disabled = value || !state.catalog;
  $("reference-button").disabled = value || !state.current;
  $("remove-reference").disabled = value;
  $("show-input").disabled = value;
  $("show-reference").disabled = value || !state.current?.reference;
  for (const button of document.querySelectorAll("[data-method]")) {
    const method = button.dataset.method;
    const live = state.catalog?.live_methods?.[method];
    button.disabled = value || !available(method);
    button.title = !state.current ? "Upload an image first."
      : available(method) ? "Run on this Mac"
      : live?.enabled ? `Use an input ${live.min_side || 16}–${live.max_side} pixels per side.`
      : live?.reason || "This method is unavailable.";
    button.setAttribute("aria-pressed", String(method === state.selected));
  }
  $("output-canvas").setAttribute("aria-busy", String(value));
}

function fileData(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({name: file.name, data: reader.result.split(",")[1]});
    reader.onerror = () => reject(new Error("Could not read the selected file."));
    reader.readAsDataURL(file);
  });
}

async function upload(file, reference = null) {
  if (!file || state.busy) return;
  error();
  if (file.size + (reference?.size || 0) > 10 * 1024 * 1024) {
    error("Please keep the combined file size below 10 MB.");
    return;
  }
  setBusy(true);
  try {
    const body = {image: await fileData(file)};
    if (reference) body.reference = await fileData(reference);
    const next = await api("api/upload", body);
    await Promise.all([next.input, next.reference].map(loadImage));
    const nextImages = new Map([next.input, next.reference].filter(Boolean).map(url => [url, images.get(url)]));
    images.clear();
    for (const [url, image] of nextImages) images.set(url, image);
    uploadFile = file;
    state.current = next;
    state.selected = null;
    state.reference = false;
    $("file-name").textContent = file.name;
    $("reference-name").textContent = reference ? reference.name : "Add the matching original for scores";
    $("remove-reference").hidden = !reference;
    render();
  } catch (e) { error(e.message); }
  finally { setBusy(false); }
}

async function chooseMethod(method) {
  if (!available(method) || state.busy) return;
  error();
  if (state.current.results[method]) {
    state.selected = method;
    render();
    return;
  }
  setBusy(true);
  $("result-foot").textContent = `Running ${names[method]}…`;
  $("empty-result").textContent = `Running ${names[method]}…`;
  try {
    const result = await api("api/run", {image_id: state.current.id, method});
    await loadImage(result.url);
    state.current.results[method] = result;
    state.selected = method;
    render();
  } catch (e) {
    error(e.message);
    $("result-foot").textContent = "Could not finish";
    $("empty-result").textContent = "Choose a method to try again.";
  } finally { setBusy(false); }
}

function score(value, digits) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function link(id, url, filename) {
  const element = $(id);
  element.hidden = !url;
  if (url) { element.href = url; element.download = filename; }
  else element.removeAttribute("href");
}

function render() {
  const c = state.current;
  const result = c?.results[state.selected];
  $("empty-input").hidden = !!c;
  $("show-reference").disabled = !c?.reference;
  $("show-input").setAttribute("aria-pressed", String(!state.reference));
  $("show-reference").setAttribute("aria-pressed", String(state.reference));
  $("input-size").textContent = c ? (state.reference ? `${c.width * 4} × ${c.height * 4}` : `${c.width} × ${c.height}`) : "";
  $("input-foot").textContent = state.reference ? "High-resolution reference" : "Low-resolution image · enlarged for viewing";
  link("download-input", state.reference ? c?.reference : c?.input, `${state.reference ? "reference" : "input"}.png`);
  $("download-input").textContent = state.reference ? "Save reference ↓" : "Save input ↓";
  $("result-title").textContent = result ? names[state.selected] : "Result";
  $("output-size").textContent = result ? `${result.width} × ${result.height}` : "";
  $("empty-result").hidden = !!result;
  $("empty-result").textContent = c ? "Choose a method above." : "Choose a method after uploading.";
  $("result-foot").textContent = result
    ? `${result.device} · ${(result.seconds * 1000).toFixed(1)} ms${result.load_seconds ? ` (+ ${result.load_seconds.toFixed(1)} s loading)` : ""}`
    : "No method selected";
  link("download-result", result?.url, `${state.selected || "output"}.png`);
  $("measurements").hidden = !c || !Object.keys(c.results).length;
  $("scores").replaceChildren(...methods.map(method => {
    const r = c?.results[method];
    const m = r?.metrics || {};
    const row = document.createElement("tr");
    row.classList.toggle("selected", state.selected === method);
    const cells = [names[method], m.psnr_infinite ? "∞" : score(m.psnr, 2), score(m.ssim, 3),
      r ? `${(r.seconds * 1000).toFixed(1)} ms` : "—"];
    cells.forEach((text, i) => {
      const cell = document.createElement(i === 0 ? "th" : "td");
      if (!i) cell.scope = "row";
      cell.textContent = text;
      row.appendChild(cell);
    });
    return row;
  }));
  $("metric-note").textContent = (c?.reference
    ? "PSNR and SSIM compare the full output with your aligned reference. Higher is better."
    : "Add a matching HR reference before running the methods to calculate PSNR and SSIM.")
    + " Inference time excludes model loading and scoring. The first run can be slower.";
  drawAll();
  setBusy(state.busy);
}

function paint(id, url, nearest = false) {
  const canvas = $(id);
  const box = canvas.getBoundingClientRect();
  if (!box.width || !box.height) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(box.width * dpr);
  canvas.height = Math.round(box.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#edece6";
  ctx.fillRect(0, 0, box.width, box.height);
  const image = images.get(url);
  if (!image) return;
  const factor = Math.min(box.width / image.width, box.height / image.height);
  const width = image.width * factor, height = image.height * factor;
  ctx.imageSmoothingEnabled = !nearest;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(image, (box.width - width) / 2, (box.height - height) / 2, width, height);
}

function drawAll() {
  const c = state.current;
  paint("input-canvas", state.reference ? c?.reference : c?.input, !state.reference);
  paint("output-canvas", c?.results[state.selected]?.url);
}

async function init() {
  document.querySelectorAll("[data-method]").forEach(button => button.addEventListener("click", () => chooseMethod(button.dataset.method)));
  $("show-input").addEventListener("click", () => { state.reference = false; render(); });
  $("show-reference").addEventListener("click", () => { state.reference = true; render(); });
  $("upload-button").addEventListener("click", () => $("upload").click());
  $("reference-button").addEventListener("click", () => $("reference-file").click());
  $("upload").addEventListener("change", async () => { await upload($("upload").files[0]); $("upload").value = ""; });
  $("reference-file").addEventListener("change", async () => {
    const file = $("reference-file").files[0];
    if (file) await upload(uploadFile, file);
    $("reference-file").value = "";
  });
  $("remove-reference").addEventListener("click", () => upload(uploadFile));
  window.addEventListener("resize", drawAll);
  render();
  try {
    state.catalog = await api("api/catalog");
    setBusy(false);
  } catch (e) { error("Could not open the demo: " + e.message); }
}

init();
