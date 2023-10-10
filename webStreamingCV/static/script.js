// Connect to the server using Socket.io
var socket = io.connect('http://' + document.domain + ':' + location.port);

// Update the user count when received from the server
socket.on('update_count', function(data) {
  var countElement = document.querySelector('.website-counter');
  countElement.textContent = data.count;
});
