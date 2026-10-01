// Hardware Profile & Smartboard Optimization (Android 14 4K Board)
const isSmartboardDevice = (/Android/i.test(navigator.userAgent) || (navigator.maxTouchPoints && navigator.maxTouchPoints > 0));
if (isSmartboardDevice) {
  document.documentElement.classList.add('smartboard-device');
}

// Global Application State
const state = {
  token: localStorage.getItem('access_token') || null,
  user: JSON.parse(localStorage.getItem('user_info') || 'null'),
  currentRole: 'CLASS',
  webcamStream: null,
  isCapturing: false,
  currentSessionId: null,
  reviewItems: {},
  allSmartboardRecords: [],
  currentFilter: 'ALL',
  searchQuery: '',
  focusLockWatchdog: null
};

// Global Chart Instances
let chartInstances = {
  staffTrend: null,
  hodDoughnut: null,
  hodBar: null
};

// API Helper
async function apiCall(endpoint, options = {}) {
  const headers = options.headers || {};
  if (state.token) {
    headers['Authorization'] = `Bearer ${state.token}`;
  }
  
  if (!(options.body instanceof FormData) && options.body && typeof options.body === 'object') {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }

  options.headers = headers;

  let response;
  try {
    response = await fetch(endpoint, options);
  } catch (netErr) {
    if (window.location.protocol === 'file:') {
      throw new Error('You opened the HTML file directly. Please access the portal at http://localhost:8000 instead.');
    }
    throw new Error('Cannot connect to backend server. Ensure the server is running on http://localhost:8000.');
  }

  if (response.status === 401 && endpoint !== '/api/auth/login') {
    if (state.token) {
      logout();
      throw new Error('Session expired. Please login again.');
    }
  }
  
  let data;
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    data = await response.json();
  } else {
    const rawText = await response.text();
    if (!response.ok) {
      throw new Error(`Server Error (${response.status}): ${rawText || response.statusText}`);
    }
    data = rawText;
  }

  if (!response.ok) {
    throw new Error(data.detail || 'API request failed');
  }
  return data;
}

// Toast Notifications
function showToast(message, type = 'success') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerText = message;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

// Autofill Login Demo
window.autofillLogin = function(username, password, role) {
  document.getElementById('loginUsername').value = username;
  document.getElementById('loginPassword').value = password;
  state.currentRole = role;
  
  document.querySelectorAll('.role-pill').forEach(pill => {
    pill.classList.toggle('active', pill.dataset.role === role);
  });
};

// Academic Particles & UI Animation Helpers
function initAcademicParticles() {
  const container = document.getElementById('academicParticles');
  if (!container) return;
  container.innerHTML = '';
  // Reduce to 6 particles on touch/Android smartboard to save GPU budget
  const isTouchDevice = (navigator.maxTouchPoints > 0);
  const particleCount = isTouchDevice ? 6 : 16;
  for (let i = 0; i < particleCount; i++) {
    const p = document.createElement('div');
    p.className = 'academic-node';
    const size = Math.floor(Math.random() * 24) + 12;
    p.style.width = `${size}px`;
    p.style.height = `${size}px`;
    p.style.top = `${Math.random() * 94}vh`;
    p.style.left = `${Math.random() * 95}vw`;
    p.style.animationDuration = `${isTouchDevice ? (Math.random() * 10 + 30) : (Math.random() * 12 + 14)}s`;
    p.style.animationDelay = `${(Math.random() * 5).toFixed(1)}s`;
    p.style.opacity = isTouchDevice ? '0.08' : (Math.random() * 0.35 + 0.15).toFixed(2);
    container.appendChild(p);
  }
}

function animateCountUp(elementId, targetValue, durationMs = 600) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const start = parseInt(el.innerText) || 0;
  const target = parseInt(targetValue) || 0;
  if (start === target) {
    el.innerText = target;
    return;
  }
  el.classList.add('pulse-count');
  setTimeout(() => el.classList.remove('pulse-count'), durationMs + 100);

  const startTime = performance.now();
  function updateNumber(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(1, elapsed / durationMs);
    const current = Math.round(start + (target - start) * (1 - (1 - progress) * (1 - progress)));
    el.innerText = current;
    if (progress < 1) {
      requestAnimationFrame(updateNumber);
    } else {
      el.innerText = target;
    }
  }
  requestAnimationFrame(updateNumber);
}

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initAcademicParticles();
  setupEventListeners();
  setupKeyboardShortcuts();
  if (state.token && state.user) {
    showAuthenticatedUI();
  } else {
    showScreen('loginScreen');
  }
});

function setupEventListeners() {
  document.querySelectorAll('.role-pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
      document.querySelectorAll('.role-pill').forEach(p => p.classList.remove('active'));
      e.target.classList.add('active');
      state.currentRole = e.target.dataset.role;
    });
  });

  document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;

    try {
      const res = await apiCall('/api/auth/login', {
        method: 'POST',
        body: { username, password }
      });

      state.token = res.access_token;
      state.user = res;
      localStorage.setItem('access_token', res.access_token);
      localStorage.setItem('user_info', JSON.stringify(res));

      showToast(`Welcome back, ${res.full_name}!`);
      showAuthenticatedUI();
    } catch (err) {
      showToast(err.message, 'error');
    }
  });

  document.getElementById('logoutBtn').addEventListener('click', logout);
  document.getElementById('startAttendanceBtn').addEventListener('click', startSmartboardAttendance);

  const sbExcelBtn = document.getElementById('sbDownloadExcelBtn');
  if (sbExcelBtn) {
    sbExcelBtn.addEventListener('click', () => downloadExcelReport(state.currentSessionId));
  }

  document.getElementById('sendForReviewBtn').addEventListener('click', () => {
    showToast('Attendance results forwarded for Staff Confirmation.');
    if (state.user.role === 'STAFF' || state.user.role === 'ADMIN') {
      showScreen('staffScreen');
      loadStaffDashboard();
    }
  });

  document.querySelectorAll('.admin-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      document.querySelectorAll('.admin-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.admin-tab-content').forEach(c => c.classList.remove('active'));
      
      e.target.classList.add('active');
      document.getElementById(e.target.dataset.tab).classList.add('active');

      if (e.target.dataset.tab === 'tabUsers') loadAdminUsers();
      if (e.target.dataset.tab === 'tabStructure') loadAdminStructure();
      if (e.target.dataset.tab === 'tabAudit') loadAdminAudit();
    });
  });

  const dropZone = document.getElementById('fileDropZone');
  const fileInput = document.getElementById('zipFileInput');

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files[0]));

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
  });

  document.getElementById('uploadZipForm').addEventListener('submit', handleZipUpload);

  document.getElementById('confirmAndGenerateExcelBtn').addEventListener('click', submitStaffConfirmation);
  document.getElementById('closeReviewCardBtn').addEventListener('click', () => {
    document.getElementById('staffReviewCard').style.display = 'none';
  });

  document.getElementById('refreshHodBtn').addEventListener('click', loadHodDashboard);
  document.getElementById('refreshStaffHistoryBtn').addEventListener('click', loadStaffDashboard);

  document.getElementById('createUserForm').addEventListener('submit', handleCreateUser);
  document.getElementById('createDeptForm').addEventListener('submit', handleCreateDept);
  document.getElementById('createClassForm').addEventListener('submit', handleCreateClass);

  const searchInput = document.getElementById('sbSearchInput');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      state.searchQuery = e.target.value;
      filterAndRenderSmartboardTable();
    });
  }

  document.querySelectorAll('.shortlist-pill').forEach(pill => {
    pill.addEventListener('click', (e) => {
      const btn = e.target.closest('.shortlist-pill');
      if (!btn) return;
      document.querySelectorAll('.shortlist-pill').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      state.currentFilter = btn.dataset.filter;
      filterAndRenderSmartboardTable();
    });
  });

  const closeModalBtn = document.getElementById('closeStudentModalBtn');
  if (closeModalBtn) closeModalBtn.addEventListener('click', closeStudentModal);

  const modalBackdrop = document.getElementById('studentDetailModal');
  if (modalBackdrop) {
    modalBackdrop.addEventListener('click', (e) => {
      if (e.target === modalBackdrop) closeStudentModal();
    });
  }
}

function setupKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    const reviewCard = document.getElementById('staffReviewCard');
    if (reviewCard && reviewCard.style.display !== 'none') {
      const keys = Object.keys(state.reviewItems);
      if (keys.length === 0) return;

      if (e.key === 'p' || e.key === 'P') {
        keys.forEach(k => setReviewDecision(parseInt(k), 'PRESENT'));
        showToast('Keyboard Shortcut: All pending marked PRESENT');
      } else if (e.key === 'a' || e.key === 'A') {
        keys.forEach(k => setReviewDecision(parseInt(k), 'ABSENT'));
        showToast('Keyboard Shortcut: All pending marked ABSENT');
      } else if (e.key === 'Enter') {
        submitStaffConfirmation();
      }
    }
  });
}

function logout() {
  stopWebcam();
  state.token = null;
  state.user = null;
  localStorage.removeItem('access_token');
  localStorage.removeItem('user_info');
  document.getElementById('mainNavbar').style.display = 'none';
  showScreen('loginScreen');
  showToast('Logged out successfully.');
}

function showScreen(screenId) {
  document.querySelectorAll('.view-screen').forEach(s => s.style.display = 'none');
  document.getElementById(screenId).style.display = 'block';
}

function showAuthenticatedUI() {
  document.getElementById('mainNavbar').style.display = 'flex';
  document.getElementById('userRoleBadge').innerText = state.user.role;
  document.getElementById('navUserName').innerText = state.user.full_name;
  document.getElementById('navUserDept').innerText = state.user.role === 'ADMIN' ? 'System Administrator' : 'Department Portal';

  const role = state.user.role;

  if (role === 'CLASS') {
    showScreen('smartboardScreen');
    initSmartboardView();
  } else if (role === 'STAFF') {
    showScreen('staffScreen');
    loadStaffDashboard();
  } else if (role === 'HOD') {
    showScreen('hodScreen');
    loadHodDashboard();
  } else if (role === 'ADMIN') {
    showScreen('adminScreen');
    loadAdminDatasetTab();
  }
}

/* ==========================================================================
   ROLE 1: CLASSROOM SMARTBOARD ATTENDANCE & LIVE CANVAS OVERLAYS
   ========================================================================== */
async function initSmartboardView() {
  const now = new Date();
  document.getElementById('sbDateDisplay').innerText = `Date: ${now.toLocaleDateString()}`;
  document.getElementById('sbTimeDisplay').innerText = `Time: ${now.toLocaleTimeString()}`;

  try {
    const classes = await apiCall('/api/admin/classes');
    if (classes.length > 0) {
      const activeClass = classes.find(c => c.student_count > 0) || classes[0];
      state.selectedClassId = activeClass.id;
      document.getElementById('sbClassDisplay').innerText = `Class: ${activeClass.name}`;
    }
    const startBtn = document.getElementById('startAttendanceBtn');
    if (startBtn && !state.currentSessionId) {
      startBtn.disabled = false;
      startBtn.classList.remove('btn-frozen', 'btn-completed');
      startBtn.classList.add('pulse-btn');
      startBtn.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        [ START ATTENDANCE SCAN ]
      `;
    }
    initWebcam();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// Camera Stabilization & Continuous Focus Engine
async function assertCameraFocusLock(videoTrack) {
  if (!videoTrack || typeof videoTrack.getCapabilities !== 'function') return false;
  try {
    const capabilities = videoTrack.getCapabilities();
    const advancedConstraints = {};
    let lockApplied = false;

    // Prefer continuous autofocus to keep all classroom rows sharp
    if (capabilities.focusMode && capabilities.focusMode.includes('continuous')) {
      advancedConstraints.focusMode = 'continuous';
      lockApplied = true;
    }
    // Prefer continuous auto-exposure
    if (capabilities.exposureMode && capabilities.exposureMode.includes('continuous')) {
      advancedConstraints.exposureMode = 'continuous';
      lockApplied = true;
    }
    // Prefer continuous auto-white-balance
    if (capabilities.whiteBalanceMode && capabilities.whiteBalanceMode.includes('continuous')) {
      advancedConstraints.whiteBalanceMode = 'continuous';
      lockApplied = true;
    }

    if (lockApplied) {
      await videoTrack.applyConstraints({ advanced: [advancedConstraints] });
      return true;
    }
  } catch (err) {
    console.warn('[Camera] Focus/exposure setup warning:', err);
  }
  return false;
}

function computeFrameSharpness(ctx, width, height) {
  try {
    const sampleW = Math.min(240, width);
    const sampleH = Math.min(135, height);
    const startX = Math.floor((width - sampleW) / 2);
    const startY = Math.floor((height - sampleH) / 2);
    
    const imgData = ctx.getImageData(startX, startY, sampleW, sampleH);
    const d = imgData.data;
    let diff = 0;
    let count = 0;

    for (let i = 0; i < d.length - 8; i += 16) {
      const l1 = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      const l2 = 0.299 * d[i + 4] + 0.587 * d[i + 5] + 0.114 * d[i + 6];
      diff += Math.abs(l1 - l2);
      count++;
    }
    return count > 0 ? (diff / count) : 50;
  } catch (e) {
    return 50;
  }
}

async function initWebcam() {
  const video = document.getElementById('webcamVideo');
  const placeholder = document.getElementById('cameraPlaceholder');
  const pill = document.getElementById('cameraStatusPill');
  const statusText = document.getElementById('cameraStatusText');

  try {
    // STAGE 1: Request optimal classroom camera resolution, supporting any camera
    // from low-budget webcams (down to 240p/280p) up to Full HD & 4K smartboards.
    const cameraConstraints = {
      video: {
        width: { ideal: 2560, min: 320 },
        height: { ideal: 1440, min: 240 },
        facingMode: 'environment'        // Prefer rear-facing / wide-angle camera
      }
    };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(cameraConstraints);
    } catch (_constraintErr) {
      console.warn('[Camera] Advanced constraints not accepted, falling back to basic camera open.');
      stream = await navigator.mediaDevices.getUserMedia({
        video: true
      });
    }

    state.webcamStream = stream;

    // STAGE 2: Force camera out of autofocus and lock focus distance
    const videoTrack = state.webcamStream.getVideoTracks()[0];
    let focusLockStatus = 'Hardware Auto-Locked (Fixed Focus)';

    if (videoTrack) {
      const locked = await assertCameraFocusLock(videoTrack);
      if (locked) {
        focusLockStatus = 'Auto-Lock Active (Autofocus Disabled)';
      }
    }

    video.srcObject = state.webcamStream;
    placeholder.style.display = 'none';
    pill.className = 'camera-status-pill ready';
    statusText.innerText = `🔒 ${focusLockStatus}`;
    console.log(`[Camera] Stream active. Focus status: ${focusLockStatus}`);

  } catch (err) {
    console.warn('Webcam hardware not found or permission denied:', err);
    placeholder.innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 1l22 22"/><path d="M21 21l-3-3m-3-3L3 3"/><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/></svg><p>Webcam not connected. Using simulated classroom smartboard mode.</p>`;
    pill.className = 'camera-status-pill';
    statusText.innerText = 'Simulated Mode';
  }
}

function stopWebcam() {
  if (state.focusLockWatchdog) {
    clearInterval(state.focusLockWatchdog);
    state.focusLockWatchdog = null;
  }
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
}

async function startSmartboardAttendance() {
  if (state.isCapturing) return;

  const startBtn = document.getElementById('startAttendanceBtn');
  if (startBtn) {
    startBtn.disabled = true;
    startBtn.classList.add('btn-frozen');
    startBtn.classList.remove('pulse-btn');
    startBtn.innerHTML = `
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="spin-icon">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"/>
      </svg>
      [ ATTENDANCE IN PROGRESS... ]
    `;
  }

  state.isCapturing = true;
  const pill = document.getElementById('cameraStatusPill');
  const statusText = document.getElementById('cameraStatusText');

  pill.className = 'camera-status-pill recording';
  statusText.innerText = '📹 Optimizing camera focus & exposure...';

  const video = document.getElementById('webcamVideo');
  const videoTrack = state.webcamStream ? state.webcamStream.getVideoTracks()[0] : null;

  if (videoTrack) {
    await assertCameraFocusLock(videoTrack);
  }

  // Capture at native video resolution (supports 240p, 280p, 360p, 480p, 720p, 1080p, or up to 2560px QHD)
  const maxDim = 2560;
  const vidW = video.videoWidth || 640;
  const vidH = video.videoHeight || 480;
  const scale = vidW > maxDim ? (maxDim / vidW) : 1.0;
  const captureWidth = Math.round(vidW * scale);
  const captureHeight = Math.round(vidH * scale);
  const canvas = document.createElement('canvas');
  canvas.width = captureWidth;
  canvas.height = captureHeight;
  // Use willReadFrequently for faster getImageData on Android GPU
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  // STEP 2: Brief stabilization — let camera exposure settle
  if (state.webcamStream && video.readyState === 4) {
    ctx.drawImage(video, 0, 0, captureWidth, captureHeight);
    await new Promise(r => setTimeout(r, 150));
  }

  statusText.innerHTML = '📹 <strong>Starting 5-Second Video Sweep...</strong>';

  const capturedFrames = [];
  // Capture 12-14 sharp keyframes across 5-second video sweep (covering all classroom rows from left to right)
  const captureDurationMs = 5000;
  const intervalMs = 380;
  const startTime = Date.now();

  const timer = setInterval(() => {
    const elapsed = Date.now() - startTime;
    const remaining = Math.max(0, (captureDurationMs - elapsed) / 1000).toFixed(1);
    const progressPct = Math.min(100, Math.round((elapsed / captureDurationMs) * 100));

    statusText.innerHTML = `📹 <span style="color:#EF4444;font-weight:800;">● RECORDING VIDEO SWEEP (${remaining}s remaining)</span> — Pan camera slowly across all students`;
    if (startBtn) {
      startBtn.innerHTML = `
        <span class="recording-pulse-dot" style="display:inline-block;width:12px;height:12px;background:#EF4444;border-radius:50%;margin-right:8px;box-shadow:0 0 8px #EF4444;"></span>
        [ 📹 RECORDING VIDEO SWEEP: ${remaining}s (${progressPct}%) ]
      `;
    }

    if (state.webcamStream && video.readyState >= 2) {
      ctx.drawImage(video, 0, 0, captureWidth, captureHeight);
      capturedFrames.push(canvas.toDataURL('image/jpeg', 0.88));
    } else {
      ctx.fillStyle = '#1E293B';
      ctx.fillRect(0, 0, captureWidth, captureHeight);
      ctx.fillStyle = '#38BDF8';
      ctx.font = '20px sans-serif';
      ctx.fillText(`Classroom Smartboard Frame ${capturedFrames.length + 1}`, 50, 100);
      capturedFrames.push(canvas.toDataURL('image/jpeg', 0.90));
    }

    if (elapsed >= captureDurationMs) {
      clearInterval(timer);
      pill.className = 'camera-status-pill ready';
      statusText.innerHTML = `⚡ <strong>Evaluating Neural Facial Biometrics (${capturedFrames.length} frames across 5s sweep)...</strong>`;
      if (startBtn) {
        startBtn.innerHTML = `
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" class="spin-icon">
            <circle cx="12" cy="12" r="10" stroke-opacity="0.25"/>
            <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"/>
          </svg>
          [ ⚡ PROCESSING 5s VIDEO SWEEP... ]
        `;
      }

      sendFramesForProcessing(capturedFrames);
    }
  }, intervalMs);
}

async function sendFramesForProcessing(frames) {
  try {
    const classes = await apiCall('/api/admin/classes');
    const activeClass = (state.selectedClassId && classes.find(c => c.id === state.selectedClassId)) ||
                        classes.find(c => c.student_count > 0) ||
                        classes[0];
    const classId = activeClass ? activeClass.id : 1;

    const res = await apiCall('/api/attendance/process-frames', {
      method: 'POST',
      body: {
        class_id: classId,
        frames: frames
      }
    });

    // Clear watchdog interval once processing finishes, but leave camera in locked state
    if (state.focusLockWatchdog) {
      clearInterval(state.focusLockWatchdog);
      state.focusLockWatchdog = null;
    }

    state.isCapturing = false;
    state.currentSessionId = res.session_id;

    // Permanently freeze the button for this attendance session
    const startBtn = document.getElementById('startAttendanceBtn');
    if (startBtn) {
      startBtn.disabled = true;
      startBtn.classList.add('btn-frozen', 'btn-completed');
      startBtn.classList.remove('pulse-btn');
      startBtn.innerHTML = `
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5">
          <polyline points="20 6 9 17 4 12"/>
        </svg>
        [ ATTENDANCE RECORDED &amp; COMPLETED ]
      `;
    }

    document.getElementById('sbSessionStatus').innerText = 'COMPLETED';
    document.getElementById('sbSessionStatus').className = 'session-badge badge-present';
    document.getElementById('cameraStatusText').innerText = '🔒 Completed (Focus Locked)';
    
    // Smooth number count-up animation
    animateCountUp('sbTotalStudents', res.total_students);
    animateCountUp('sbPresentCount', res.present_count);
    animateCountUp('sbPendingCount', res.pending_count);
    animateCountUp('sbAbsentCount', res.absent_count);

    const cAll = document.getElementById('countAll'); if (cAll) animateCountUp('countAll', res.total_students);
    const cPres = document.getElementById('countPresent'); if (cPres) animateCountUp('countPresent', res.present_count);
    const cRev = document.getElementById('countReview'); if (cRev) animateCountUp('countReview', res.pending_count);
    const cAbs = document.getElementById('countAbsent'); if (cAbs) animateCountUp('countAbsent', res.absent_count);

    state.allSmartboardRecords = res.records || [];
    filterAndRenderSmartboardTable();

    if (res.frame_overlays && res.frame_overlays.length > 0) {
      renderLiveCanvasOverlays(res.frame_overlays);
    }

    document.getElementById('sbActionFooter').style.display = 'block';
    showToast(`Attendance Complete! ${res.present_count} Present, ${res.absent_count} Absent of ${res.total_students} students.`);
  } catch (err) {
    if (state.focusLockWatchdog) {
      clearInterval(state.focusLockWatchdog);
      state.focusLockWatchdog = null;
    }
    state.isCapturing = false;
    showToast(err.message, 'error');
    // Restore button in case of failure so user can retry
    const startBtn = document.getElementById('startAttendanceBtn');
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.classList.remove('btn-frozen', 'btn-completed');
      startBtn.classList.add('pulse-btn');
      startBtn.innerHTML = `
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        [ START ATTENDANCE SCAN ]
      `;
    }
  }
}

function filterAndRenderSmartboardTable() {
  if (!state.allSmartboardRecords) return;

  const q = (state.searchQuery || '').toLowerCase().trim();
  const filter = state.currentFilter || 'ALL';

  const filtered = state.allSmartboardRecords.filter(r => {
    const matchSearch = !q || r.student_id.toLowerCase().includes(q) || r.student_name.toLowerCase().includes(q);
    const matchFilter = (filter === 'ALL') || (r.status === filter);
    return matchSearch && matchFilter;
  });

  renderSmartboardResultsTable(filtered);
}

function renderLiveCanvasOverlays(frameOverlays) {
  const canvas = document.getElementById('webcamCanvas');
  const ctx = canvas.getContext('2d');
  const video = document.getElementById('webcamVideo');

  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Animate bounding boxes over captured frames
  let idx = 0;
  const anim = setInterval(() => {
    if (idx >= frameOverlays.length) {
      clearInterval(anim);
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const item = frameOverlays[idx];

    item.boxes.forEach(b => {
      const [x, y, w, h] = b.box;
      
      // Draw Bounding Box
      ctx.strokeStyle = b.score >= 45 ? '#10B981' : '#F59E0B';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // Draw Label Badge
      ctx.fillStyle = b.score >= 45 ? '#10B981' : '#F59E0B';
      ctx.fillRect(x, y - 24, ctx.measureText(b.label).width + 16, 24);
      ctx.fillStyle = '#000000';
      ctx.font = 'bold 12px sans-serif';
      ctx.fillText(b.label, x + 8, y - 7);
    });

    idx++;
  }, 150);
}

function renderSmartboardResultsTable(records) {
  const tbody = document.getElementById('sbResultsTableBody');
  tbody.innerHTML = '';

  records.forEach((r, idx) => {
    const tr = document.createElement('tr');
    tr.className = 'cascade-row';
    tr.style.animationDelay = `${Math.min(0.6, idx * 0.02)}s`;
    let badgeClass = 'badge-absent';
    if (r.status === 'PRESENT') badgeClass = 'badge-present';
    if (r.status === 'REVIEW') badgeClass = 'badge-review';

    const confDisplay = r.confidence ? (String(r.confidence).endsWith('%') ? r.confidence : `${r.confidence}%`) : '0%';

    tr.innerHTML = `
      <td><strong>${r.student_id}</strong></td>
      <td>${r.student_name}</td>
      <td>${confDisplay}</td>
      <td>
        <button type="button" class="badge ${badgeClass} status-toggle-btn" 
          onclick="toggleStudentAttendanceStatus('${r.student_id}', '${r.status}')" 
          title="Click to toggle Present/Absent" style="cursor: pointer; border: 1px solid rgba(255,255,255,0.15); font-size: 0.82rem; padding: 4px 10px; border-radius: 6px; transition: transform 0.15s ease;">
          ${r.status} ✎
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

window.toggleStudentAttendanceStatus = async function(studentId, currentStatus) {
  if (!state.currentSessionId) {
    showToast('No active attendance session recorded yet.', 'warning');
    return;
  }
  const newStatus = (currentStatus === 'PRESENT') ? 'ABSENT' : 'PRESENT';
  
  try {
    const res = await apiCall('/api/attendance/update-student-status', {
      method: 'POST',
      body: {
        session_id: state.currentSessionId,
        student_id: studentId,
        status: newStatus
      }
    });

    if (state.allSmartboardRecords) {
      const rec = state.allSmartboardRecords.find(r => r.student_id === studentId);
      if (rec) {
        rec.status = newStatus;
        if (newStatus === 'PRESENT') {
          rec.confidence = Math.max(rec.confidence || 0, 95.0);
        }
      }
      filterAndRenderSmartboardTable();
    }

    if (res.present_count !== undefined) {
      document.getElementById('sbPresentCount').innerText = res.present_count;
      document.getElementById('sbAbsentCount').innerText = res.absent_count;
      document.getElementById('sbPendingCount').innerText = res.pending_count;
      const cPres = document.getElementById('countPresent'); if (cPres) cPres.innerText = res.present_count;
      const cAbs = document.getElementById('countAbsent'); if (cAbs) cAbs.innerText = res.absent_count;
      const cRev = document.getElementById('countReview'); if (cRev) cRev.innerText = res.pending_count;
    }

    showToast(`Updated: Student ${studentId} marked ${newStatus}`);
  } catch (err) {
    showToast(err.message || 'Failed to update student status', 'error');
  }
};

/* ==========================================================================
   ROLE 2: STAFF DASHBOARD & REVIEW CONFIRMATION
   ========================================================================== */
async function loadStaffDashboard() {
  try {
    const history = await apiCall('/api/attendance/history');
    const tbody = document.getElementById('staffSessionsTableBody');
    tbody.innerHTML = '';

    let pendingCountTotal = 0;

    history.forEach(s => {
      if (s.pending > 0) pendingCountTotal += s.pending;

      const tr = document.createElement('tr');
      let statusBadge = `<span class="badge badge-present">CONFIRMED</span>`;
      let actionBtn = `<button class="btn btn-outline btn-sm" onclick="downloadExcelReport(${s.id})">📥 Download Excel</button>`;

      if (s.status === 'PENDING_REVIEW' || s.pending > 0) {
        statusBadge = `<span class="badge badge-review">NEEDS CONFIRMATION</span>`;
        actionBtn = `<button class="btn btn-warning btn-sm" onclick="openStaffReviewModal(${s.id})">🔍 Review & Confirm</button>`;
      }

      tr.innerHTML = `
        <td><strong>${s.class_name}</strong></td>
        <td>${s.date} ${s.time}</td>
        <td>${s.total_students}</td>
        <td><span class="text-success">${s.present}</span></td>
        <td><span class="text-warning"><strong>${s.pending}</strong></span></td>
        <td><span class="text-danger">${s.absent}</span></td>
        <td>${statusBadge}</td>
        <td>${actionBtn}</td>
      `;
      tbody.appendChild(tr);
    });

    const alertBox = document.getElementById('staffPendingAlert');
    alertBox.style.display = pendingCountTotal > 0 ? 'flex' : 'none';

    renderStaffTrendChart(history);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function openStaffReviewModal(sessionId) {
  try {
    const data = await apiCall(`/api/attendance/session/${sessionId}`);
    state.currentSessionId = sessionId;
    state.reviewItems = {};

    document.getElementById('reviewSessionId').innerText = sessionId;
    const list = document.getElementById('reviewItemsList');
    list.innerHTML = '';

    if (data.reviews.length === 0) {
      list.innerHTML = `<div class="text-muted p-3 text-center">No uncertain faces pending confirmation in this session.</div>`;
    } else {
      data.reviews.forEach(rv => {
        state.reviewItems[rv.student_db_id] = 'PRESENT';

        const div = document.createElement('div');
        div.className = 'review-item-card';
        div.innerHTML = `
          <div class="review-compare-box">
            <div class="photo-cmp">
              <img src="${rv.captured_crop || rv.registered_photo}" alt="Captured" />
              <label>Captured Face (Webcam)</label>
            </div>
            <div class="photo-cmp">
              <img src="${rv.registered_photo}" alt="Registered" />
              <label>Registered Photo</label>
            </div>
            <div class="review-info">
              <h4>${rv.student_id} — ${rv.student_name}</h4>
              <p>Recognition Confidence: <strong>${rv.match_score}%</strong> (UNCERTAIN)</p>
              <p>Status: <span class="badge badge-review" id="badge_st_${rv.student_db_id}">PENDING STAFF DECISION</span></p>
              <small class="text-muted">Press 'P' for Present, 'A' for Absent, 'Enter' to Confirm</small>
            </div>
          </div>
          <div class="review-decision-btns">
            <button class="btn btn-success" id="btn_pres_${rv.student_db_id}" onclick="setReviewDecision(${rv.student_db_id}, 'PRESENT')">
              [ MARK PRESENT ] (P)
            </button>
            <button class="btn btn-outline" id="btn_abs_${rv.student_db_id}" onclick="setReviewDecision(${rv.student_db_id}, 'ABSENT')">
              [ MARK ABSENT ] (A)
            </button>
          </div>
        `;
        list.appendChild(div);
      });
    }

    document.getElementById('staffReviewCard').style.display = 'block';
    window.scrollTo({ top: document.getElementById('staffReviewCard').offsetTop - 80, behavior: 'smooth' });
  } catch (err) {
    showToast(err.message, 'error');
  }
}

window.setReviewDecision = function(studentDbId, status) {
  state.reviewItems[studentDbId] = status;

  const btnPres = document.getElementById(`btn_pres_${studentDbId}`);
  const btnAbs = document.getElementById(`btn_abs_${studentDbId}`);
  const badge = document.getElementById(`badge_st_${studentDbId}`);

  if (status === 'PRESENT') {
    btnPres.className = 'btn btn-success';
    btnAbs.className = 'btn btn-outline';
    badge.className = 'badge badge-present';
    badge.innerText = 'DECIDED: PRESENT';
  } else {
    btnPres.className = 'btn btn-outline';
    btnAbs.className = 'btn btn-danger';
    badge.className = 'badge badge-absent';
    badge.innerText = 'DECIDED: ABSENT';
  }
};

async function submitStaffConfirmation() {
  if (!state.currentSessionId) return;

  const confirmations = Object.keys(state.reviewItems).map(stId => ({
    student_id: parseInt(stId),
    status: state.reviewItems[stId]
  }));

  try {
    await apiCall('/api/attendance/confirm-review', {
      method: 'POST',
      body: {
        session_id: state.currentSessionId,
        confirmations: confirmations
      }
    });

    showToast('Staff confirmation saved. Generating final Excel attendance sheet...');
    document.getElementById('staffReviewCard').style.display = 'none';

    await downloadExcelReport(state.currentSessionId);
    loadStaffDashboard();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function downloadExcelReport(sessionId) {
  if (!sessionId) {
    showToast('No active attendance session to download.', 'error');
    return;
  }

  const modal = document.getElementById('downloadModal');
  const bar = document.getElementById('dlProgressFill');
  const pct = document.getElementById('dlPctText');
  const step = document.getElementById('dlStepText');
  const title = document.getElementById('dlModalTitle');
  const fname = document.getElementById('dlModalFilename');
  const check = document.getElementById('dlSuccessBadge');

  if (modal) {
    modal.style.display = 'flex';
    if (bar) bar.style.width = '18%';
    if (pct) pct.innerText = '18%';
    if (step) step.innerText = '🏛️ Connecting to V.S.B. Academic Attendance Registry...';
    if (title) title.innerText = 'PREPARING ATTENDANCE EXCEL REPORT';
    if (fname) fname.innerText = `VSB_Attendance_Session_${sessionId}.xlsx`;
    if (check) check.classList.remove('show');
  }

  try {
    if (bar) bar.style.width = '42%';
    if (pct) pct.innerText = '42%';
    if (step) step.innerText = '📊 Compiling Student Present / Absent Roll Matrix...';

    const res = await apiCall(`/api/attendance/generate-excel/${sessionId}`, { method: 'POST' });
    
    if (fname && res.file_name) fname.innerText = res.file_name;
    if (bar) bar.style.width = '78%';
    if (pct) pct.innerText = '78%';
    if (step) step.innerText = '🔐 Applying Cryptographic Institutional Timestamps & Formatting .xlsx...';

    const response = await fetch(res.download_url, {
      headers: {
        'Authorization': `Bearer ${state.token}`
      }
    });

    if (!response.ok) {
      throw new Error('Failed to retrieve Excel file');
    }

    if (bar) bar.style.width = '100%';
    if (pct) pct.innerText = '100%';
    if (step) step.innerText = '✅ Report Ready! Saving .xlsx Spreadsheet to Device...';
    if (check) check.classList.add('show');

    const blob = await response.blob();
    const blobUrl = window.URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = res.file_name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(blobUrl);

    showToast(`Excel Attendance Report ${res.file_name} downloaded!`);

    setTimeout(() => {
      if (modal) modal.style.display = 'none';
      if (check) check.classList.remove('show');
    }, 750);
  } catch (err) {
    if (modal) modal.style.display = 'none';
    showToast(err.message, 'error');
  }
}

/* ==========================================================================
   ROLE 3: HOD DASHBOARD
   ========================================================================== */
async function loadHodDashboard() {
  try {
    const dateVal = document.getElementById('hodDateFilter').value;
    const url = dateVal ? `/api/hod/dashboard?date=${dateVal}` : '/api/hod/dashboard';
    const res = await apiCall(url);

    document.getElementById('hodDeptTitle').innerText = `${res.department_name} (${res.department_code})`;
    document.getElementById('hodTotalClasses').innerText = res.total_classes;
    document.getElementById('hodTotalStudents').innerText = res.total_students;
    document.getElementById('hodDeptPresent').innerText = res.dept_present;
    document.getElementById('hodOverallPct').innerText = res.overall_percentage;

    const tbody = document.getElementById('hodClassTableBody');
    tbody.innerHTML = '';

    res.classes.forEach(c => {
      const tr = document.createElement('tr');
      const downloadBtn = c.session_id
        ? `<button class="btn btn-outline btn-sm" onclick="downloadExcelReport(${c.session_id})">📥 Download Excel</button>`
        : `<span class="text-muted">No Session</span>`;

      tr.innerHTML = `
        <td><strong>${c.class_name}</strong></td>
        <td>${c.total_students}</td>
        <td><span class="text-success">${c.present}</span></td>
        <td><span class="text-danger">${c.absent}</span></td>
        <td><strong>${c.percentage}</strong></td>
        <td><span class="badge ${c.status === 'CONFIRMED' ? 'badge-present' : 'badge-review'}">${c.status}</span></td>
        <td>${downloadBtn}</td>
      `;
      tbody.appendChild(tr);
    });

    renderHodCharts(res);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

/* ==========================================================================
   ROLE 4: ADMIN PORTAL & ZIP DATASET UPLOAD
   ========================================================================== */
async function loadAdminDatasetTab() {
  try {
    const classes = await apiCall('/api/admin/classes');
    const select = document.getElementById('adminZipClassSelect');
    select.innerHTML = '';
    classes.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.innerText = `${c.name} (${c.department_name})`;
      select.appendChild(opt);
    });

    if (classes.length > 0) loadStudentRoster(classes[0].id);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function handleFileSelect(file) {
  if (!file) return;
  state.selectedZipFile = file;
  document.getElementById('zipFileNameDisplay').innerHTML = `Selected File: <strong>${file.name}</strong> (${(file.size/1024/1024).toFixed(2)} MB)`;
}

async function handleZipUpload(e) {
  e.preventDefault();
  if (!state.selectedZipFile) {
    showToast('Please select a student dataset .ZIP file.', 'error');
    return;
  }

  const classId = document.getElementById('adminZipClassSelect').value;
  const formData = new FormData();
  formData.append('class_id', classId);
  formData.append('file', state.selectedZipFile);

  const btn = document.getElementById('zipUploadSubmitBtn');
  btn.disabled = true;
  btn.innerText = 'Extracting & Computing Face Embeddings...';

  try {
    const res = await apiCall('/api/admin/upload-zip-dataset', {
      method: 'POST',
      body: formData
    });

    document.getElementById('ingestionReportEmpty').style.display = 'none';
    document.getElementById('ingestionReportContent').style.display = 'block';

    document.getElementById('repTotal').innerText = res.total_students;
    document.getElementById('repSuccess').innerText = res.successfully_processed;
    document.getElementById('repInvalid').innerText = res.invalid_images + res.multiple_face_images + res.no_face_images + res.poor_quality_images;

    document.getElementById('repNoFace').innerText = res.no_face_images;
    document.getElementById('repMultiFace').innerText = res.multiple_face_images;
    document.getElementById('repBlurry').innerText = res.poor_quality_images;
    document.getElementById('repDuplicates').innerText = res.duplicate_ids;

    const logBox = document.getElementById('reportLogBox');
    logBox.innerHTML = res.details.map(d => `[${d.status}] ${d.filename} (${d.student_id}) — ${d.message}`).join('<br>');

    showToast(`Dataset Processed! ${res.successfully_processed} student facial embeddings stored.`);
    loadStudentRoster(classId);
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerText = 'Process & Extract Facial Embeddings';
  }
}

window.handleReindexBiometrics = async function() {
  const btn = document.getElementById('reindexBiometricsBtn');
  if (btn) {
    btn.disabled = true;
    btn.innerText = '⚡ Computing 10-Template Biometric Neural Profiles...';
  }

  try {
    const classSelect = document.getElementById('adminZipClassSelect');
    const classId = classSelect ? classSelect.value : null;
    const url = classId ? `/api/admin/reindex-embeddings?class_id=${classId}` : '/api/admin/reindex-embeddings';
    const res = await apiCall(url, { method: 'POST' });

    showToast(`Biometrics Upgraded! ${res.successfully_reindexed} students re-indexed with ${res.total_templates_generated} neural templates.`);
    if (classId) {
      loadStudentRoster(classId);
    }
  } catch (err) {
    showToast(err.message || 'Failed to re-index biometric profiles.', 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = '⚡ Re-Index 10-Template Deep Biometrics for All Enrolled Students';
    }
  }
};

async function loadStudentRoster(classId) {
  try {
    const students = await apiCall(`/api/admin/students?class_id=${classId}`);
    const grid = document.getElementById('studentRosterGrid');
    grid.innerHTML = '';

    students.forEach(st => {
      const card = document.createElement('div');
      card.className = 'student-card';
      card.innerHTML = `
        <img src="${st.photo_path || '/static/assets/avatar.png'}" alt="${st.name}" />
        <div class="st-id">${st.student_id}</div>
        <div class="st-name">${st.name}</div>
      `;
      card.addEventListener('click', () => openStudentModal(st));
      grid.appendChild(card);
    });
  } catch (err) {
    console.error(err);
  }
}

async function loadAdminUsers() {
  try {
    const users = await apiCall('/api/admin/users');
    const tbody = document.getElementById('usersTableBody');
    tbody.innerHTML = '';
    users.forEach(u => {
      const tr = document.createElement('tr');
      tr.innerHTML = `<td><strong>${u.username}</strong></td><td>${u.full_name}</td><td>${u.role}</td><td>${u.department_name || 'All'}</td>`;
      tbody.appendChild(tr);
    });

    const depts = await apiCall('/api/admin/departments');
    const sel = document.getElementById('newUserDeptSelect');
    sel.innerHTML = '<option value="">None / System Wide</option>';
    depts.forEach(d => sel.innerHTML += `<option value="${d.id}">${d.name}</option>`);
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function loadAdminStructure() {
  try {
    const depts = await apiCall('/api/admin/departments');
    const classes = await apiCall('/api/admin/classes');

    const dList = document.getElementById('deptListGroup');
    dList.innerHTML = depts.map(d => `<li class="p-2 border-bottom"><strong>${d.name}</strong> (${d.code})</li>`).join('');

    const cSel = document.getElementById('classDeptSelect');
    cSel.innerHTML = depts.map(d => `<option value="${d.id}">${d.name}</option>`).join('');

    const cList = document.getElementById('classListGroup');
    cList.innerHTML = classes.map(c => `<li class="p-2 border-bottom"><strong>${c.name}</strong> (${c.department_name})</li>`).join('');

    const sSel = document.getElementById('subjectClassSelect');
    sSel.innerHTML = classes.map(c => `<option value="${c.id}">${c.name}</option>`).join('');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function loadAdminAudit() {
  try {
    const logs = await apiCall('/api/admin/audit-logs');
    const tbody = document.getElementById('auditLogsTableBody');
    tbody.innerHTML = logs.map(l => `
      <tr>
        <td>${new Date(l.timestamp).toLocaleString()}</td>
        <td><strong>${l.username || 'System'}</strong></td>
        <td>${l.action}</td>
        <td>${l.details}</td>
      </tr>
    `).join('');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleCreateUser(e) {
  e.preventDefault();
  try {
    await apiCall('/api/admin/users', {
      method: 'POST',
      body: {
        username: document.getElementById('newUsername').value,
        password: document.getElementById('newPassword').value,
        full_name: document.getElementById('newFullName').value,
        role: document.getElementById('newRole').value,
        department_id: document.getElementById('newUserDeptSelect').value || null
      }
    });
    showToast('User created successfully!');
    loadAdminUsers();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleCreateDept(e) {
  e.preventDefault();
  try {
    await apiCall('/api/admin/departments', {
      method: 'POST',
      body: {
        name: document.getElementById('deptName').value,
        code: document.getElementById('deptCode').value
      }
    });
    showToast('Department added!');
    loadAdminStructure();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function handleCreateClass(e) {
  e.preventDefault();
  try {
    await apiCall('/api/admin/classes', {
      method: 'POST',
      body: {
        name: document.getElementById('classNameInput').value,
        department_id: parseInt(document.getElementById('classDeptSelect').value)
      }
    });
    showToast('Class added!');
    loadAdminStructure();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

/* ==========================================================================
   ANALYTICS CHARTS & STUDENT MODAL HELPERS
   ========================================================================== */
function renderStaffTrendChart(sessions) {
  const ctx = document.getElementById('staffTrendChart')?.getContext('2d');
  if (!ctx || typeof Chart === 'undefined') return;

  if (chartInstances.staffTrend) {
    chartInstances.staffTrend.destroy();
  }

  const labels = sessions.slice(0, 10).reverse().map(s => `${s.class_name} (${s.date})`);
  const dataPresent = sessions.slice(0, 10).reverse().map(s => (s.total_students ? Math.round((s.present / s.total_students) * 100) : 0));

  chartInstances.staffTrend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels.length ? labels : ['No Sessions'],
      datasets: [{
        label: 'Attendance Rate (%)',
        data: dataPresent.length ? dataPresent : [0],
        borderColor: '#38BDF8',
        backgroundColor: 'rgba(56, 189, 248, 0.15)',
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#38BDF8',
        pointRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#9CA3AF' } }
      },
      scales: {
        x: { ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { min: 0, max: 100, ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

function renderHodCharts(res) {
  const doughnutCtx = document.getElementById('hodAttendanceChart')?.getContext('2d');
  const barCtx = document.getElementById('hodClassBarChart')?.getContext('2d');
  if (typeof Chart === 'undefined') return;

  if (doughnutCtx) {
    if (chartInstances.hodDoughnut) chartInstances.hodDoughnut.destroy();
    const presentCount = res.dept_present || 0;
    const absentCount = Math.max(0, (res.total_students || 0) - presentCount);

    chartInstances.hodDoughnut = new Chart(doughnutCtx, {
      type: 'doughnut',
      data: {
        labels: ['Present', 'Absent'],
        datasets: [{
          data: [presentCount, absentCount],
          backgroundColor: ['#10B981', '#EF4444'],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#F9FAFB', font: { weight: 'bold' } } }
        }
      }
    });
  }

  if (barCtx) {
    if (chartInstances.hodBar) chartInstances.hodBar.destroy();
    const classLabels = res.classes.map(c => c.class_name);
    const classPcts = res.classes.map(c => parseFloat(c.percentage.replace('%', '')) || 0);

    chartInstances.hodBar = new Chart(barCtx, {
      type: 'bar',
      data: {
        labels: classLabels.length ? classLabels : ['No Classes'],
        datasets: [{
          label: 'Attendance %',
          data: classPcts.length ? classPcts : [0],
          backgroundColor: 'rgba(139, 92, 246, 0.7)',
          borderColor: '#8B5CF6',
          borderWidth: 1,
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#9CA3AF' }, grid: { display: false } },
          y: { min: 0, max: 100, ticks: { color: '#9CA3AF' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });
  }
}

function openStudentModal(st) {
  const modal = document.getElementById('studentDetailModal');
  if (!modal) return;
  document.getElementById('modalStudentName').innerText = st.name;
  document.getElementById('modalStudentFullName').innerText = st.name;
  document.getElementById('modalStudentRoll').innerText = `Roll #${st.student_id}`;
  document.getElementById('modalStudentPhoto').src = st.photo_path || '/static/assets/avatar.png';
  document.getElementById('modalStudentDeptClass').innerText = `Class: ${st.class_name || 'III AIDS A'}`;
  modal.style.display = 'flex';
}

function closeStudentModal() {
  const modal = document.getElementById('studentDetailModal');
  if (modal) modal.style.display = 'none';
}

