/* ============================================================
   admin.js — QAD-MultiGuard 管理看板共享工具
   ============================================================ */

const API = '/admin/api';
const V1  = '/v1/admin';

/* ── Auth check ─────────────────────────────────────── */
async function checkAuth() {
  try {
    const res = await fetch(API + '/me');
    if (!res.ok) throw new Error('not authed');
    const data = await res.json();
    document.querySelectorAll('.sidebar-user .name').forEach(el => {
      el.textContent = data.nickname || data.phone;
    });
    return data;
  } catch (e) {
    if (!location.pathname.includes('/login')) {
      location.href = '/admin/login';
    }
    return null;
  }
}

/* ── HTTP helpers ───────────────────────────────────── */
async function apiGet(url) {
  const res = await fetch(url, { credentials: 'same-origin' });
  if (res.status === 401) { location.href = '/admin/login'; return null; }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || '请求失败');
  }
  return res.json();
}

async function apiPost(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(body),
  });
  if (res.status === 401) { location.href = '/admin/login'; return null; }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || '请求失败');
  }
  return res.json();
}

async function apiPut(url, body) {
  const res = await fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify(body),
  });
  if (res.status === 401) { location.href = '/admin/login'; return null; }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || '请求失败');
  }
  return res.json();
}

async function apiDelete(url) {
  const res = await fetch(url, {
    method: 'DELETE',
    credentials: 'same-origin',
  });
  if (res.status === 401) { location.href = '/admin/login'; return null; }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || '请求失败');
  }
  return res.json();
}

/* ── Toast ──────────────────────────────────────────── */
function showToast(message, type = 'info') {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => { toast.remove(); }, 3000);
}

/* ── Pagination ─────────────────────────────────────── */
function renderPagination(container, meta, fetchFn) {
  if (!meta || meta.total === 0) {
    container.innerHTML = '<p style="text-align:center;color:var(--text-secondary);padding:20px;">暂无数据</p>';
    return;
  }
  const totalPages = Math.ceil(meta.total / (meta.limit || 10));
  const current = meta.page || 1;
  let html = '';
  html += `<button ${current <= 1 ? 'disabled' : ''} onclick="void(0)" data-page="${current - 1}">上一页</button>`;
  html += `<span class="current">${current} / ${totalPages}</span>`;
  html += `<button ${current >= totalPages ? 'disabled' : ''} onclick="void(0)" data-page="${current + 1}">下一页</button>`;
  html += `<span style="margin-left:8px;font-size:13px;color:var(--text-secondary);">共 ${meta.total} 条</span>`;
  container.innerHTML = html;
  container.querySelectorAll('button').forEach(btn => {
    btn.addEventListener('click', () => {
      const page = parseInt(btn.dataset.page);
      if (!isNaN(page) && page >= 1) fetchFn(page);
    });
  });
}

/* ── Risk badge ─────────────────────────────────────── */
function riskBadge(level) {
  const map = { high: 'badge-high', medium: 'badge-medium', safe: 'badge-safe' };
  const labels = { high: '高危', medium: '中危', safe: '安全' };
  const cls = map[level] || 'badge-safe';
  const label = labels[level] || level;
  return `<span class="badge ${cls}">${label}</span>`;
}

function statusBadge(status) {
  const map = {
    pending: 'badge-pending', approved: 'badge-approved', rejected: 'badge-rejected',
    published: 'badge-approved', archived: 'badge-rejected', draft: 'badge-pending'
  };
  const labels = {
    pending: '待审核', approved: '已通过', rejected: '已拒绝',
    published: '已发布', archived: '已下架', draft: '草稿'
  };
  return `<span class="badge ${map[status] || 'badge-pending'}">${labels[status] || status}</span>`;
}

/* ── Format date ────────────────────────────────────── */
function fmtTime(iso) {
  if (!iso) return '-';
  const d = new Date(iso);
  const pad = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/* ── Modal ──────────────────────────────────────────── */
function showModal(title, content, onConfirm, confirmText = '确认') {
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.innerHTML = `
    <div class="modal">
      <h2>${title}</h2>
      <div class="modal-body">${content}</div>
      <div class="modal-actions">
        <button class="btn btn-outline close-modal">取消</button>
        <button class="btn btn-primary confirm-modal">${confirmText}</button>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  overlay.querySelector('.close-modal').onclick = () => overlay.remove();
  overlay.querySelector('.confirm-modal').onclick = async () => {
    await onConfirm(overlay);
    overlay.remove();
  };
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.remove();
  });
}

/* ── Logout ─────────────────────────────────────────── */
async function doLogout() {
  await apiPost(API + '/logout');
  location.href = '/admin/login';
}

/* ── Init on page load ──────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  // Highlight active nav
  const path = location.pathname;
  document.querySelectorAll('.sidebar-nav a').forEach(a => {
    if (path.includes(a.getAttribute('href'))) {
      a.classList.add('active');
    }
  });

  // Logout button
  document.querySelectorAll('.sidebar-user .logout').forEach(btn => {
    btn.addEventListener('click', doLogout);
  });
});
