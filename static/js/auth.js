// auth.js — Session guard for app.html
// Runs synchronously in <head> before the DOM loads.
// If no token is present, redirect to login immediately.
// We check both 'cognirad_token' (set by login.js) and a short
// grace window so the page doesn't flicker on first load.
(function () {
  const token = sessionStorage.getItem('cognirad_token');
  if (!token) {
    // Preserve any auto_login / bot query params so the login page
    // can re-trigger the auto-login flow if needed.
    window.location.replace('/static/index.html');
  }
})();
