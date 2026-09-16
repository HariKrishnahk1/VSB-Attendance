// Global Application State
const state = {
  token: localStorage.getItem('access_token') || null,
  user: JSON.parse(localStorage.getItem('user_info') || 'null'),
  currentRole: 'CLASS',
  webcamStream: null,
  isCapturing: false,
  currentSessionId: null,
  reviewItems: {}
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

  const response = await fetch(endpoint, options);
  if (response.status === 401) {
    logout();
    throw new Error('Session expired. Please login again.');
  }
  
  const data = await response.json();
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

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
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
  document.getElementById('createSubjectForm').addEventListener('submit', handleCreateSubject);

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
      document.getElementById('sbClassDisplay').innerText = `Class: ${classes[0].name}`;
      loadSubjectsForClass(classes[0].id);
    }
    initWebcam();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function loadSubjectsForClass(classId) {
  try {
    const subjects = await apiCall(`/api/admin/subjects?class_id=${classId}`);
    const select = document.getElementById('sbSubjectSelect');
    select.innerHTML = '';
    subjects.forEach(sub => {
      const opt = document.createElement('option');
      opt.value = sub.id;
      opt.innerText = `${sub.name} (${sub.code})`;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error(err);
  }
}

async function initWebcam() {
  const video = document.getElementById('webcamVideo');
  const placeholder = document.getElementById('cameraPlaceholder');
  const pill = document.getElementById('cameraStatusPill');
  const statusText = document.getElementById('cameraStatusText');

  try {
    state.webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
    });
    video.srcObject = state.webcamStream;
    placeholder.style.display = 'none';
    pill.className = 'camera-status-pill ready';
    statusText.innerText = 'Ready';
  } catch (err) {
    console.warn('Webcam hardware not found or permission denied:', err);
    placeholder.innerHTML = `<svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 1l22 22"/><path d="M21 21l-3-3m-3-3L3 3"/><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/></svg><p>Webcam not connected. Using simulated classroom smartboard mode.</p>`;
    pill.className = 'camera-status-pill';
    statusText.innerText = 'Simulated';
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
}

async function startSmartboardAttendance() {
  if (state.isCapturing) return;

  const subjectId = document.getElementById('sbSubjectSelect').value;
  if (!subjectId) {
    showToast('Please select a subject first.', 'error');
    return;
  }

  state.isCapturing = true;
  const overlay = document.getElementById('scanProgressOverlay');
  const fill = document.getElementById('scanProgressFill');
  const pill = document.getElementById('cameraStatusPill');
  const statusText = document.getElementById('cameraStatusText');

  overlay.style.display = 'flex';
  pill.className = 'camera-status-pill recording';
  statusText.innerText = 'Capturing 3.5s...';

  const video = document.getElementById('webcamVideo');
  const canvas = document.createElement('canvas');
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext('2d');

  const capturedFrames = [];
  const captureDurationMs = 3500;
  const intervalMs = 200;
  const startTime = Date.now();

  const timer = setInterval(() => {
    const elapsed = Date.now() - startTime;
    const progressPct = Math.min(100, (elapsed / captureDurationMs) * 100);
    fill.style.width = `${progressPct}%`;
    document.getElementById('scanSeconds').innerText = ((captureDurationMs - elapsed) / 1000).toFixed(1);

    if (state.webcamStream && video.readyState === 4) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      capturedFrames.push(canvas.toDataURL('image/jpeg', 0.8));
    } else {
      ctx.fillStyle = '#1E293B';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#38BDF8';
      ctx.font = '20px sans-serif';
      ctx.fillText(`Classroom Smartboard Frame ${capturedFrames.length + 1}`, 50, 100);
      capturedFrames.push(canvas.toDataURL('image/jpeg', 0.8));
    }

    if (elapsed >= captureDurationMs) {
      clearInterval(timer);
      overlay.style.display = 'none';
      pill.className = 'camera-status-pill ready';
      statusText.innerText = 'Processing...';

      sendFramesForProcessing(subjectId, capturedFrames);
    }
  }, intervalMs);
}

async function sendFramesForProcessing(subjectId, frames) {
  try {
    const classes = await apiCall('/api/admin/classes');
    const classId = classes[0].id;

    const res = await apiCall('/api/attendance/process-frames', {
      method: 'POST',
      body: {
        class_id: classId,
        subject_id: parseInt(subjectId),
        frames: frames
      }
    });

    state.isCapturing = false;
    state.currentSessionId = res.session_id;

    document.getElementById('sbSessionStatus').innerText = 'COMPLETED';
    document.getElementById('sbSessionStatus').className = 'session-badge badge-present';
    document.getElementById('sbTotalStudents').innerText = res.total_students;
    document.getElementById('sbPresentCount').innerText = res.present_count;
    document.getElementById('sbPendingCount').innerText = res.pending_count;
    document.getElementById('sbAbsentCount').innerText = res.absent_count;

    renderSmartboardResultsTable(res.records);
    if (res.frame_overlays && res.frame_overlays.length > 0) {
      renderLiveCanvasOverlays(res.frame_overlays);
    }

    document.getElementById('sbActionFooter').style.display = 'block';
    showToast(`Attendance Processed! ${res.present_count} Recognized, ${res.pending_count} Pending Review.`);
  } catch (err) {
    state.isCapturing = false;
    showToast(err.message, 'error');
  }
}

function renderLiveCanvasOverlays(frameOverlays) {
  const canvas = document.getElementById('webcamCanvas');
  const ctx = canvas.getContext('2d');
  const video = document.getElementById('webcamVideo');

  canvas.width = video.clientWidth || 640;
  canvas.height = video.clientHeight || 480;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Animate bounding boxes over captured frames
  let idx = 0;
  const overlayTimer = setInterval(() => {
    if (idx >= frameOverlays.length) {
      clearInterval(overlayTimer);
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

  records.forEach(r => {
    const tr = document.createElement('tr');
    let badgeClass = 'badge-absent';
    if (r.status === 'PRESENT') badgeClass = 'badge-present';
    if (r.status === 'REVIEW') badgeClass = 'badge-review';

    tr.innerHTML = `
      <td><strong>${r.student_id}</strong></td>
      <td>${r.student_name}</td>
      <td>${r.confidence}</td>
      <td><span class="badge ${badgeClass}">${r.status}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

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
        <td>${s.subject_name}</td>
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
  try {
    const res = await apiCall(`/api/attendance/generate-excel/${sessionId}`, { method: 'POST' });
    
    const link = document.createElement('a');
    link.href = res.download_url;
    link.download = res.file_name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showToast(`Excel Attendance Report ${res.file_name} downloaded!`);
  } catch (err) {
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
      tr.innerHTML = `
        <td><strong>${c.class_name}</strong></td>
        <td>${c.total_students}</td>
        <td><span class="text-success">${c.present}</span></td>
        <td><span class="text-danger">${c.absent}</span></td>
        <td><strong>${c.percentage}</strong></td>
        <td><span class="badge ${c.status === 'CONFIRMED' ? 'badge-present' : 'badge-review'}">${c.status}</span></td>
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

async function handleCreateSubject(e) {
  e.preventDefault();
  try {
    await apiCall('/api/admin/subjects', {
      method: 'POST',
      body: {
        name: document.getElementById('subjectNameInput').value,
        code: document.getElementById('subjectCodeInput').value,
        class_id: parseInt(document.getElementById('subjectClassSelect').value)
      }
    });
    showToast('Subject added!');
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

