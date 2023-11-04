// Connect to the server using Socket.io
var socket = io.connect('http://' + document.domain + ':' + location.port);

// Update the user count when received from the server
socket.on('update_count', function(data) {
  var countElement = document.querySelector('.website-counter');
  countElement.textContent = data.count;
});

socket.on('connect', function() {
  // Request the app uptime when connected to the server
  socket.emit('get_app_uptime');
});

socket.on('update_app_uptime', function(uptime) {
  var uptimeElement = document.getElementById('app-uptime');
  uptimeElement.textContent = uptime;
});
