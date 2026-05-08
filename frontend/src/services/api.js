import { API_BASE_URL } from '../config/api';
const BASE_URL = API_BASE_URL;

// Register User
export async function registerUser(data) {
  const res = await fetch(`${BASE_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (res.ok) return res;

  const errData = await res.json().catch(() => ({ error: "Unknown error" }));
  const err = new Error("Request failed");
  err.response = { data: errData, status: res.status };
  throw err;
}

// Login User
export async function loginUser(data) {
  const res = await fetch(`${BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (res.ok) return res;

  const errData = await res.json().catch(() => ({ error: "Unknown error" }));
  const err = new Error("Request failed");
  err.response = { data: errData, status: res.status };
  throw err;
}

// Upload PDF
export async function uploadPDF(formData) {
  const res = await fetch(`${BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (res.ok) return await res.json().catch(() => ({}));

  const errData = await res.json().catch(() => ({ error: "Unknown error" }));
  const err = new Error("Upload failed");
  err.response = { data: errData, status: res.status };
  throw err;
}

export default { registerUser, loginUser, uploadPDF };
