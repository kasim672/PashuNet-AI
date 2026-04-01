// ============================================================
// BHARAT PASHUDHAN v2 — FRONTEND LOGIC
// All API endpoints preserved: /health, /predict_single, /predict_multi
// ============================================================

// ─── STATE ─────────────────────────────────────────────────────
const state = {
  mode: 'single',
  selectedFiles: [],
  apiConnected: false,
  totalScans: 0,
  theme: localStorage.getItem('bp_theme') || 'dark'
};

// ─── DOM CACHE ──────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const els = {
  // Mode
  singleModeBtn:       $('singleModeBtn'),
  multipleModeBtn:     $('multipleModeBtn'),
  // Upload
  dropzone:            $('dropzone'),
  fileInput:           $('fileInput'),
  browseBtn:           $('browseBtn'),
  uploadSection:       $('uploadSection'),
  previewSection:      $('previewSection'),
  previewGrid:         $('previewGrid'),
  imageCount:          $('imageCount'),
  clearAllBtn:         $('clearAllBtn'),
  analyzeBtn:          $('analyzeBtn'),
  aggregationSelector: $('aggregationSelector'),
  // Loading
  loadingSection:      $('loadingSection'),
  step1: $('step1'), step2: $('step2'), step3: $('step3'),
  // Results
  resultsSection:      $('resultsSection'),
  animalTypeBadge:     $('animalTypeBadge'),
  animalTypeText:      $('animalTypeText'),
  decisionBadge:       $('decisionBadge'),
  decisionText:        $('decisionText'),
  decisionMessage:     $('decisionMessage'),
  recommendationText:  $('recommendationText'),
  predictionsList:     $('predictionsList'),
  topConfidenceBadge:  $('topConfidenceBadge'),
  breedInfoCard:       $('breedInfoCard'),
  breedInfoGrid:       $('breedInfoGrid'),
  reasoningText:       $('reasoningText'),
  newScanBtn:          $('newScanBtn'),
  // Error
  errorSection:        $('errorSection'),
  errorMessage:        $('errorMessage'),
  retryBtn:            $('retryBtn'),
  // Header
  apiStatus:           $('apiStatus'),
  totalScans:          $('totalScans'),
  // Theme
  themeToggle:         $('themeToggle'),
  themeIcon:           $('themeIcon'),
  // Toast
  toast:               $('toast'),
  toastMessage:        $('toastMessage'),
  toastIcon:           $('toastIcon'),
};

// ─── INIT ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  applyTheme();
  restoreStats();
  setupEventListeners();
  checkAPIConnection();
});

function restoreStats() {
  state.totalScans = parseInt(localStorage.getItem('bp_totalScans') || '0');
  els.totalScans.textContent = state.totalScans;
}

// ─── EVENT LISTENERS ────────────────────────────────────────────
function setupEventListeners() {
  // Mode buttons
  els.singleModeBtn.addEventListener('click',   () => setMode('single'));
  els.multipleModeBtn.addEventListener('click', () => setMode('multiple'));

  // Browse button
  els.browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    els.fileInput.click();
  });
  els.fileInput.addEventListener('change', handleFileSelect);

  // Dropzone clicks (excluding the Browse button area)
  els.dropzone.addEventListener('click', (e) => {
    if (!e.target.closest('.btn-upload')) els.fileInput.click();
  });

  // Drag & drop
  els.dropzone.addEventListener('dragover',   handleDragOver);
  els.dropzone.addEventListener('dragleave',  handleDragLeave);
  els.dropzone.addEventListener('drop',       handleDrop);

  // Keyboard accessibility for dropzone
  els.dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); els.fileInput.click(); }
  });

  // Actions
  els.clearAllBtn.addEventListener('click', clearAllFiles);
  els.analyzeBtn.addEventListener('click',  analyzeImages);
  els.newScanBtn.addEventListener('click',  resetApp);
  els.retryBtn.addEventListener('click',    analyzeImages);

  // Theme
  els.themeToggle.addEventListener('click', toggleTheme);
}

// ─── API STATUS ─────────────────────────────────────────────────
async function checkAPIConnection() {
  const dot   = els.apiStatus.querySelector('.api-dot');
  const label = els.apiStatus.querySelector('.api-label');

  try {
    const res  = await fetch('/health');
    const data = await res.json();

    if (data.status === 'healthy') {
      state.apiConnected = true;
      dot.classList.add('connected');
      dot.classList.remove('error');
      label.textContent = 'Connected';
    } else {
      throw new Error('Unhealthy');
    }
  } catch {
    state.apiConnected = false;
    dot.classList.add('error');
    label.textContent = 'Disconnected';
  }
}

// ─── MODE ────────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;

  els.singleModeBtn.classList.toggle('active',   mode === 'single');
  els.multipleModeBtn.classList.toggle('active', mode === 'multiple');
  els.singleModeBtn.setAttribute('aria-selected',   String(mode === 'single'));
  els.multipleModeBtn.setAttribute('aria-selected', String(mode === 'multiple'));

  els.fileInput.multiple                     = mode === 'multiple';
  els.aggregationSelector.style.display      = mode === 'multiple' ? 'block' : 'none';

  clearAllFiles();
}

// ─── DRAG & DROP ─────────────────────────────────────────────────
function handleDragOver(e) {
  e.preventDefault();
  els.dropzone.classList.add('drag-over');
}
function handleDragLeave(e) {
  if (!els.dropzone.contains(e.relatedTarget)) {
    els.dropzone.classList.remove('drag-over');
  }
}
function handleDrop(e) {
  e.preventDefault();
  els.dropzone.classList.remove('drag-over');
  processFiles(Array.from(e.dataTransfer.files));
}
function handleFileSelect(e) {
  processFiles(Array.from(e.target.files));
}

// ─── FILE PROCESSING ─────────────────────────────────────────────
function processFiles(files) {
  const valid = files.filter(f => {
    if (!['image/jpeg','image/jpg','image/png'].includes(f.type)) {
      showToast('Invalid type. Please use JPG or PNG.', 'error');
      return false;
    }
    if (f.size > 10 * 1024 * 1024) {
      showToast(`"${f.name}" exceeds 10 MB limit.`, 'error');
      return false;
    }
    return true;
  });
  if (!valid.length) return;

  if (state.mode === 'single' && valid.length > 1) {
    showToast('Single mode: only the first image will be used.', 'warning');
    state.selectedFiles = [valid[0]];
  } else if (state.mode === 'multiple' && valid.length > 10) {
    showToast('Maximum 10 images allowed. First 10 used.', 'warning');
    state.selectedFiles = valid.slice(0, 10);
  } else {
    state.selectedFiles = valid;
  }

  renderPreviews();
}

function renderPreviews() {
  if (!state.selectedFiles.length) {
    els.previewSection.style.display = 'none';
    return;
  }

  els.previewSection.style.display = 'block';
  els.imageCount.textContent        = state.selectedFiles.length;
  els.previewGrid.innerHTML         = '';

  state.selectedFiles.forEach((file, i) => {
    const reader = new FileReader();
    reader.onload = ({ target }) => {
      const div = document.createElement('div');
      div.className = 'preview-item';
      div.innerHTML = `
        <img src="${target.result}" alt="${file.name}" loading="lazy">
        <button class="preview-remove" onclick="removeFile(${i})" aria-label="Remove ${file.name}">
          <i class="fas fa-times"></i>
        </button>`;
      els.previewGrid.appendChild(div);
    };
    reader.readAsDataURL(file);
  });
}

function removeFile(index) {
  state.selectedFiles.splice(index, 1);
  renderPreviews();
  showToast('Image removed', 'info');
}
window.removeFile = removeFile;   // global for inline onclick

function clearAllFiles() {
  state.selectedFiles = [];
  els.fileInput.value = '';
  renderPreviews();
}

// ─── ANALYZE ────────────────────────────────────────────────────
async function analyzeImages() {
  if (!state.selectedFiles.length) {
    showToast('Please select at least one image.', 'error');
    return;
  }
  if (!state.apiConnected) {
    showToast('API is not connected. Please check the server.', 'error');
    return;
  }

  // Show loading, hide others
  setSection('loading');
  animateLoadingSteps();

  try {
    let result;
    if (state.mode === 'single') {
      result = await predictSingle(state.selectedFiles[0]);
    } else {
      const agg = document.querySelector('input[name="aggregation"]:checked').value;
      result = await predictMultiple(state.selectedFiles, agg);
    }

    renderResults(result);

    // Persist scan count
    state.totalScans++;
    localStorage.setItem('bp_totalScans', state.totalScans);
    els.totalScans.textContent = state.totalScans;

    showToast('Analysis complete!', 'success');

  } catch (err) {
    showError(err.message || 'Prediction failed. Please try again.');
  }
}

// ─── API CALLS ───────────────────────────────────────────────────
async function predictSingle(file) {
  const fd = new FormData();
  fd.append('file', file);

  const res = await fetch('/predict_single', { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`Server error ${res.status}. Please try again.`);
  return res.json();
}

async function predictMultiple(files, aggregation) {
  const fd = new FormData();
  files.forEach(f => fd.append('files', f));
  fd.append('aggregation', aggregation);

  const res = await fetch('/predict_multi', { method: 'POST', body: fd });
  if (!res.ok) throw new Error(`Server error ${res.status}. Please try again.`);
  return res.json();
}

// ─── LOADING STEP ANIMATION ──────────────────────────────────────
function animateLoadingSteps() {
  [els.step1, els.step2, els.step3].forEach(s => s.classList.remove('active'));
  els.step1.classList.add('active');
  setTimeout(() => els.step2.classList.add('active'), 600);
  setTimeout(() => els.step3.classList.add('active'), 1400);
}

// ─── RENDER RESULTS ──────────────────────────────────────────────
function renderResults(data) {
  setSection('results');

  // Animal type
  els.animalTypeText.textContent = data.animal_type.toUpperCase();

  // Decision
  const decision = (data.decision || '').toLowerCase();
  els.decisionBadge.className  = `decision-badge ${decision}`;
  els.decisionText.textContent = data.decision;
  const decisionIconMap = { accepted: 'fa-check-circle', review: 'fa-exclamation-circle', rejected: 'fa-times-circle' };
  els.decisionBadge.querySelector('i').className = `fas ${decisionIconMap[decision] || 'fa-info-circle'}`;
  els.decisionMessage.textContent  = data.decision_message  || '';
  els.recommendationText.textContent = data.recommendation  || '';

  // Top confidence
  els.topConfidenceBadge.textContent = data.confidence_percent || '—';

  // Predictions
  els.predictionsList.innerHTML = '';
  (data.top_predictions || []).forEach(pred => {
    const el = document.createElement('div');
    el.className = 'prediction-item';
    el.innerHTML = `
      <div class="prediction-rank">${pred.rank}</div>
      <div class="prediction-info">
        <div class="prediction-name">${pred.breed}</div>
        <div class="confidence-bar">
          <div class="confidence-fill" style="width:0%" data-target="${(pred.confidence * 100).toFixed(1)}%"></div>
        </div>
      </div>
      <div class="prediction-confidence">${pred.confidence_percent}</div>`;
    els.predictionsList.appendChild(el);
  });

  // Animate confidence bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.confidence-fill').forEach(bar => {
      setTimeout(() => { bar.style.width = bar.dataset.target; }, 80);
    });
  });

  // Breed info
  if (data.breed_info) {
    els.breedInfoCard.style.display = 'block';
    renderBreedInfo(data.breed_info);
  } else {
    els.breedInfoCard.style.display = 'none';
  }

  // Reasoning
  els.reasoningText.textContent = data.reasoning || '';
}

function renderBreedInfo(info) {
  els.breedInfoGrid.innerHTML = '';
  Object.entries(info).forEach(([key, value]) => {
    const el = document.createElement('div');
    el.className = 'breed-info-item';
    el.innerHTML = `<strong>${key.replace(/_/g, ' ').toUpperCase()}</strong><span>${value}</span>`;
    els.breedInfoGrid.appendChild(el);
  });
}

// ─── ERROR DISPLAY ───────────────────────────────────────────────
function showError(message) {
  setSection('error');
  els.errorMessage.textContent = message;
}

// ─── SECTION SWITCHER ────────────────────────────────────────────
function setSection(active) {
  els.uploadSection.style.display  = active === 'upload'  ? 'block'  : 'none';
  els.loadingSection.style.display = active === 'loading' ? 'block'  : 'none';
  els.resultsSection.style.display = active === 'results' ? 'block'  : 'none';
  els.errorSection.style.display   = active === 'error'   ? 'block'  : 'none';
}

function resetApp() {
  clearAllFiles();
  setSection('upload');
  els.uploadSection.style.display = 'block';
}

// ─── THEME ───────────────────────────────────────────────────────
function toggleTheme() {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  localStorage.setItem('bp_theme', state.theme);
  applyTheme();
}
function applyTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  els.themeIcon.className = state.theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

// ─── TOAST ───────────────────────────────────────────────────────
let _toastTimer = null;
function showToast(message, type = 'success') {
  clearTimeout(_toastTimer);

  els.toastMessage.textContent = message;
  els.toastIcon.className      = `toast-icon ${type}`;

  const iconMap = { success: 'fa-check-circle', error: 'fa-times-circle', warning: 'fa-exclamation-circle', info: 'fa-info-circle' };
  els.toastIcon.innerHTML = `<i class="fas ${iconMap[type] || iconMap.info}"></i>`;

  els.toast.style.display = 'flex';
  _toastTimer = setTimeout(() => { els.toast.style.display = 'none'; }, 3500);
}
