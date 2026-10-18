// Thin wrapper around the SmartSpray simulator API (same origin as the page).

async function call(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new Error(`${path}: ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => call('GET', '/health'),
  newSession: (seed, preset) => call('POST', '/sim/session', { seed, preset }),
  dropMarker: (id, markerId, x, y) => call('POST', `/sim/${id}/marker`, { marker_id: markerId, x_m: x, y_m: y }),
  overview: (id) => call('POST', `/sim/${id}/overview`),
  capture: (id, targetId, x, y, gradcam) =>
    call('POST', `/sim/${id}/capture`, { target_id: targetId, drone_x_m: x, drone_y_m: y, gradcam }),
  plan: (id) => call('POST', `/sim/${id}/plan`),
  replan: (id, position, tankMl, tankTreatment, remaining) =>
    call('POST', `/sim/${id}/replan`, { position, tank_ml: tankMl, tank_treatment: tankTreatment, remaining }),
  finish: (id, telemetry) => call('POST', `/sim/${id}/finish`, { telemetry }),
  markerUrl: (markerId) => `/sim/aruco/${markerId}.png`,
  fileUrl: (id, name) => `/sim/${id}/files/${name}`,
  waypointUrl: (id, sortie) => `/drone/mission/${id}/${sortie}.waypoints`,
};
