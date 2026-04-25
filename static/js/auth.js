// auth.js - Blocking guard to prevent unauthorized access
const token = sessionStorage.getItem('cognirad_token');
if (!token) {
    window.location.replace('/static/index.html');
}
